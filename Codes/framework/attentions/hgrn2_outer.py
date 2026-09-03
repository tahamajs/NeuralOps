"""HGRN-2 Outer-Product Recurrent Linear Attention Module.

Implements multi-head bidirectional outer-product associative matrix memory updates
with log-space bounded decay gating, denominator normalization, and pretrained weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_outer")
class HGRN2OuterBlock(BaseAttention):
    r"""HGRN-2 Outer-Product Recurrent Linear Attention Module.

    Mathematical Mechanism:
        For multi-head queries $Q \in \mathbb{R}^{B \times H \times L \times d_k}$, keys $K$, values $V$:
        1. **Feature Normalization & Bounded Forget Gating:**
           $$\tilde{q}_t = \text{RMSNorm}(q_t), \quad \tilde{k}_t = \text{RMSNorm}(k_t) / \sqrt{d_k}$$
           $$f_t = \exp(-\text{softplus}(W_f x_t + b_f)) \in (0, 1)^{B \times H}$$
        2. **Bidirectional Associative Matrix Recurrence:**
           $$\vec{S}_t = f_t \odot \vec{S}_{t-1} + (1 - f_t) \odot (\tilde{k}_t \otimes v_t)$$
           $$\vec{z}_t = f_t \odot \vec{z}_{t-1} + (1 - f_t) \odot \tilde{k}_t$$
           $$\overleftarrow{S}_t = f_t \odot \overleftarrow{S}_{t+1} + (1 - f_t) \odot (\tilde{k}_t \otimes v_t)$$
           $$\overleftarrow{z}_t = f_t \odot \overleftarrow{z}_{t+1} + (1 - f_t) \odot \tilde{k}_t$$
        3. **Normalized Readout & Output:**
           $$\vec{y}_t = \frac{\vec{S}_t \tilde{q}_t}{\vec{z}_t^\top \tilde{q}_t + \epsilon}, \quad \overleftarrow{y}_t = \frac{\overleftarrow{S}_t \tilde{q}_t}{\overleftarrow{z}_t^\top \tilde{q}_t + \epsilon}$$
           $$\text{Output} = \text{Concat}\left(\frac{1}{2}(\vec{Y} + \overleftarrow{Y})\right) W_{\text{out}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$ linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent state matrix memory.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        
        # Resolve dim
        if dim is None:
            if isinstance(config, int):
                dim = config
            elif config is not None:
                dim = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))))
        if dim is None:
            dim = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 128)))

        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        # Resolve num_heads
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim: int = int(dim)
        self.num_heads: int = int(num_heads)
        
        # Ensure num_heads divides dim
        while self.num_heads > 1 and (self.dim % self.num_heads != 0):
            self.num_heads -= 1
            
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64

        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.W_f = nn.Linear(self.dim, self.num_heads, bias=True)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)

        # Initialize forget gate bias to -3.0 (~0.95 retention factor)
        nn.init.constant_(self.W_f.bias, -3.0)

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained Multi-Head Attention weights."""
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

    def _outer_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel outer-product attention scan (zero Python loops)."""
        B, H, L, d = q.shape

        # Log f for continuous retention decay
        log_f = torch.log(f.clamp(min=1e-4, max=1.0 - 1e-4))  # [B, H, L]
        C = torch.cumsum(log_f, dim=-1)                       # [B, H, L]
        C_diff = C.unsqueeze(-1) - C.unsqueeze(-2)            # [B, H, L, L]

        # 1. Forward causal decay matrix
        tril_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_causal = torch.exp(C_diff.clamp(min=-40.0, max=0.0)) * tril_mask.float()  # [B, H, L, L]

        # 2. Backward anticausal decay matrix
        triu_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_anti = torch.exp((-C_diff).clamp(min=-40.0, max=0.0)) * triu_mask.float()  # [B, H, L, L]

        attn_scores = (q @ k.transpose(-1, -2))  # [B, H, L, L]
        weight_in = (1.0 - f).unsqueeze(-1)      # [B, H, L, 1]
        v_scaled = v * weight_in                 # [B, H, L, d]

        if mask is not None:
            mask_float = mask.float() if mask.dim() == 2 else mask[:, 0, 0, :].float()
            mask_2d = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(1).unsqueeze(-2)
            D_causal = D_causal * mask_2d
            D_anti = D_anti * mask_2d

        # Numerator: (attn_scores * D) @ v_scaled
        num_fwd = (attn_scores * D_causal) @ v_scaled  # [B, H, L, d]
        num_bwd = (attn_scores * D_anti) @ v_scaled    # [B, H, L, d]

        # Denominator: (attn_scores * D) @ (1 - f)
        denom_fwd = (attn_scores * D_causal) @ weight_in  # [B, H, L, 1]
        denom_bwd = (attn_scores * D_anti) @ weight_in    # [B, H, L, 1]

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
        """Forward pass for HGRN2OuterBlock."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # 1. Projections & normalizations
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, d).transpose(1, 2)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1) / (d ** 0.5)

        # 2. Forget gate in (0, 1)
        f = torch.exp(-F.softplus(self.W_f(hidden_states))).transpose(1, 2)  # [B, H, L]

        # 3. Mask processing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        # 4. Parallel Bidirectional Outer Product Recurrence
        if self.training and q.requires_grad:
            y_fwd, y_bwd = torch.utils.checkpoint.checkpoint(
                self._outer_scan_parallel, q, k, v, f, mask_bool, use_reentrant=False
            )
        else:
            y_fwd, y_bwd = self._outer_scan_parallel(q, k, v, f, mask_bool)

        y = 0.5 * (y_fwd + y_bwd)  # [B, H, L, d]

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(y)

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("hgrn2_outer_inject")
def inject_hgrn2_outer(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> HGRN2OuterBlock:
    r"""Injection hook for HGRN2OuterBlock."""
    dim = 768
    num_heads = 12
    if model_config is not None:
        dim = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        num_heads = getattr(model_config, "num_attention_heads", 12)
    elif hasattr(layer, "config"):
        dim = getattr(layer.config, "hidden_size", 768)
        num_heads = getattr(layer.config, "num_attention_heads", 12)

    new_block = HGRN2OuterBlock(dim=dim, num_heads=num_heads, config=model_config, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
