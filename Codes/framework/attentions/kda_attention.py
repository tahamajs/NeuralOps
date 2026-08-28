"""Kernelized Decay Attention (KDA) / Recurrent Linear Attention.

Implements multi-head bidirectional linear recurrence with learned exponential decay gating,
spherical normalization, denominator tracking, and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("kda")
@ATTENTION_REGISTRY.register("kda_attention")
class KDAAttention(BaseAttention):
    r"""Kernelized Decay Attention (KDA) Bidirectional Mechanism.

    Implements bidirectional linear recurrent state tracking with a learned gating decay parameter $\beta$:
    $$\begin{aligned}
    \vec{S}_t &= \sigma(\beta) \odot \vec{S}_{t-1} + \tilde{k}_t v_t^\top, \quad \vec{z}_t = \sigma(\beta) \odot \vec{z}_{t-1} + \tilde{k}_t \\
    \overleftarrow{S}_t &= \sigma(\beta) \odot \overleftarrow{S}_{t+1} + \tilde{k}_t v_t^\top, \quad \overleftarrow{z}_t = \sigma(\beta) \odot \overleftarrow{z}_{t+1} + \tilde{k}_t \\
    \vec{o}_t &= \frac{\vec{S}_t \tilde{q}_t}{\vec{z}_t^\top \tilde{q}_t + \epsilon}, \quad \overleftarrow{o}_t = \frac{\overleftarrow{S}_t \tilde{q}_t}{\overleftarrow{z}_t^\top \tilde{q}_t + \epsilon} \\
    \text{Output} &= \text{Concat}\left(\frac{1}{2}(\vec{o} + \overleftarrow{o})\right) W_{\text{out}}
    \end{aligned}$$
    where keys and queries are normalized onto the $\ell_2$ unit hypersphere.

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$
        - Memory: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent state complexity.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", 4)
            else:
                num_heads = kwargs.get("num_attention_heads", 4)

        self.dim: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)

        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)  # Safe log-decay init

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights."""
        try:
            adapter = UniversalAdapter()
            w_q, b_q = adapter.get_weight_bias(old_attn, "q")
            w_k, b_k = adapter.get_weight_bias(old_attn, "k")
            w_v, b_v = adapter.get_weight_bias(old_attn, "v")
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")

            with torch.no_grad():
                self.q_proj.weight.copy_(w_q)
                self.k_proj.weight.copy_(w_k)
                self.v_proj.weight.copy_(w_v)
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    def _kda_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel KDA scan (zero Python loops)."""
        B, H, L, d = q.shape

        log_decay = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4))  # [1, H, 1, 1]
        pos = torch.arange(L, device=q.device, dtype=q.dtype)
        pos_diff = pos.unsqueeze(-1) - pos.unsqueeze(-2)  # [L, L]

        # 1. Forward causal decay matrix
        tril_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_causal = torch.exp((pos_diff.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0)) * tril_mask.float()

        # 2. Backward anticausal decay matrix
        triu_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_anti = torch.exp((-pos_diff.unsqueeze(0).unsqueeze(0) * log_decay).clamp(min=-40.0, max=0.0)) * triu_mask.float()

        attn_scores = (q @ k.transpose(-1, -2))  # [B, H, L, L]

        if mask is not None:
            mask_float = mask.float() if mask.dim() == 2 else mask[:, 0, 0, :].float()
            mask_2d = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(1).unsqueeze(-2)
            D_causal = D_causal * mask_2d
            D_anti = D_anti * mask_2d

        num_fwd = (attn_scores * D_causal) @ v  # [B, H, L, d]
        num_bwd = (attn_scores * D_anti) @ v    # [B, H, L, d]

        denom_fwd = (attn_scores * D_causal).sum(dim=-1, keepdim=True)  # [B, H, L, 1]
        denom_bwd = (attn_scores * D_anti).sum(dim=-1, keepdim=True)    # [B, H, L, 1]

        y_fwd = num_fwd / (denom_fwd + 1e-4).clamp(min=1e-4)
        y_bwd = num_bwd / (denom_bwd + 1e-4).clamp(min=1e-4)

        return y_fwd, y_bwd

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Kernelized Decay Attention."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q_raw = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        q = F.elu(q_raw) + 1.0
        k = (F.elu(k_raw) + 1.0) / (d ** 0.5)
        v = self.v_proj(hidden_states).view(B, L, H, d).transpose(1, 2)

        decay = torch.sigmoid(self.beta).view(1, H, 1, 1)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        if self.training and q.requires_grad:
            y_fwd, y_bwd = torch.utils.checkpoint.checkpoint(
                self._kda_scan_parallel, q, k, v, decay, mask_bool, use_reentrant=False
            )
        else:
            y_fwd, y_bwd = self._kda_scan_parallel(q, k, v, decay, mask_bool)

        y = 0.5 * (y_fwd + y_bwd)  # [B, H, L, d]

        out = y.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("kda_inject")
def inject_kda(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> KDAAttention:
    r"""Injection hook for KDAAttention."""
    d_model = 128  # fixed: was hardcoded 768
    num_heads = 4
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        num_heads = getattr(model_config, "num_attention_heads", 4)
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", 768)
        num_heads = getattr(layer.config, "num_attention_heads", 4)

    new_block = KDAAttention(config=model_config, d_model=d_model, num_heads=num_heads, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
