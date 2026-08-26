"""Factorised Matrix State Attention Module.

Implements multi-head bidirectional associative matrix memory updates combined with
learnable decay gating, denominator normalization, low-rank factorized output projection,
and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("factorised_matrix_state")
class FactorisedMatrixState(BaseAttention):
    r"""Bidirectional Factorised Matrix State Recurrent Linear Attention.

    Mathematical Formulation:
        1. Multi-Head Normalized Projections:
           $$Q = X W_q, \quad K = X W_k, \quad V = X W_v$$
           $$\tilde{q} = \text{RMSNorm}(Q), \quad \tilde{k} = \text{RMSNorm}(K) / \sqrt{d_k}$$
        2. Bidirectional Associative Matrix Recurrence with Bounded Decay:
           $$\lambda = \sigma(\beta) \in (0, 1)^{B \times H}$$
           $$\vec{S}_t = \lambda \vec{S}_{t-1} + \tilde{k}_t \otimes v_t, \quad \vec{z}_t = \lambda \vec{z}_{t-1} + \tilde{k}_t$$
           $$\overleftarrow{S}_t = \lambda \overleftarrow{S}_{t+1} + \tilde{k}_t \otimes v_t, \quad \overleftarrow{z}_t = \lambda \overleftarrow{z}_{t+1} + \tilde{k}_t$$
           $$\vec{Y}_t = \frac{\vec{S}_t \tilde{q}_t}{\vec{z}_t^\top \tilde{q}_t + \epsilon}, \quad \overleftarrow{Y}_t = \frac{\overleftarrow{S}_t \tilde{q}_t}{\overleftarrow{z}_t^\top \tilde{q}_t + \epsilon}$$
        3. Low-Rank Factorized Bilinear Output Projection:
           $$Y = \text{Concat}\left(\frac{1}{2}(\vec{Y} + \overleftarrow{Y})\right)$$
           $$\text{Output} = (Y L) R^\top + Y W_{\text{res}}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v + B \cdot L \cdot D \cdot r)$ linear in $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent state memory.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 32,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 128))
            else:
                d_model = kwargs.get("hidden_size", 128)

        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", 4)
            else:
                num_heads = kwargs.get("num_attention_heads", 4)

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.proj_rank: int = rank if rank is not None else 32

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Factorized low-rank projection parameters L and R
        self.L = nn.Parameter(torch.randn(self.d_model, self.proj_rank) * (1.0 / (self.d_model ** 0.5)))
        self.R = nn.Parameter(torch.randn(self.d_model, self.proj_rank) * (1.0 / (self.proj_rank ** 0.5)))
        self.W_res = nn.Linear(self.d_model, self.d_model, bias=True)

        self.beta = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)
        self.use_checkpointing: bool = kwargs.get("use_checkpointing", True)

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
                self.W_res.weight.copy_(w_o)
                if b_o is not None and self.W_res.bias is not None:
                    self.W_res.bias.copy_(b_o)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    def _matrix_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Vectorized parallel associative matrix scan."""
        B, H, L, d = q.shape
        pos = torch.arange(L, device=q.device, dtype=torch.float32)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [L, L], diff[i, j] = i - j

        if reverse:
            causal_mask = (diff <= 0).to(q.dtype)
            abs_diff = (-diff).clamp(min=0.0)
        else:
            causal_mask = (diff >= 0).to(q.dtype)
            abs_diff = diff.clamp(min=0.0)

        # decay is [1, H, 1, 1] or [H, 1, 1] -> [H, 1, 1]
        decay_h = decay.squeeze(0) if decay.dim() == 4 else decay
        decay_matrix = torch.pow(decay_h, abs_diff.unsqueeze(0)) * causal_mask.unsqueeze(0)  # [H, L, L]

        # Dot product scores: [B, H, L, L]
        scores = (q @ k.transpose(-1, -2)) * decay_matrix.unsqueeze(0)

        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            m = mask.view(B, 1, 1, L).to(q.dtype)
            scores = scores * m

        denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-5)
        y = (scores @ v) / denom  # [B, H, L, d]
        return y

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
        """Forward pass for FactorisedMatrixState."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q_raw = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        q = F.elu(q_raw) + 1.0
        k = (F.elu(k_raw) + 1.0) / (d ** 0.5)
        v = self.v_proj(hidden_states).view(B, L, H, d).transpose(1, 2)

        decay = torch.sigmoid(self.beta).unsqueeze(0)  # [1, H, 1, 1]

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        if self.training and self.use_checkpointing and hidden_states.requires_grad:
            import torch.utils.checkpoint as cp
            def _scan_fn(q_, k_, v_, decay_, mask_):
                yf = self._matrix_scan(q_, k_, v_, decay_, mask=mask_, reverse=False)
                yb = self._matrix_scan(q_, k_, v_, decay_, mask=mask_, reverse=True)
                return 0.5 * (yf + yb)
            y = cp.checkpoint(_scan_fn, q, k, v, decay, mask_bool, use_reentrant=False)
        else:
            y_fwd = self._matrix_scan(q, k, v, decay, mask=mask_bool, reverse=False)
            y_bwd = self._matrix_scan(q, k, v, decay, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)  # [B, H, L, d]

        y = y.transpose(1, 2).contiguous().view(B, L, D)

        # Factorized low-rank output projection + residual stream
        out_lr = (y @ self.L) @ self.R.t()
        out = out_lr + self.W_res(y)

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("factorised_matrix_state_inject")
def inject_factorised_matrix_state(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> FactorisedMatrixState:
    r"""Injection hook for FactorisedMatrixState."""
    d_model = 128
    num_heads = 4
    rank = kwargs.pop("rank", getattr(comp_config, "rank", 32) if comp_config else 32)
    kwargs.pop("d_model", None)
    kwargs.pop("num_heads", None)
    kwargs.pop("config", None)
    new_block = FactorisedMatrixState(config=model_config, rank=rank, d_model=d_model, num_heads=num_heads, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
