r"""Gated DeltaNet Attention Module.

Implements multi-head bidirectional Delta Rule linear recurrent attention with
input-dependent exponential decay gating, associative state error correction, and
pretrained self-attention weight transfer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gated_delta_net")
class GatedDeltaNet(BaseAttention):
    r"""Bidirectional Gated DeltaNet Recurrent Linear Attention.

    Mathematical Formulation:
        For input sequence $X \in \mathbb{R}^{B \times L \times D}$:
        1. Multi-head Projections & Normalization:
           $$Q = X W_q, \quad K = X W_k, \quad V = X W_v, \quad Z = X W_z$$
           $$\tilde{q} = \text{RMSNorm}(Q), \quad \tilde{k} = \text{RMSNorm}(K) / \sqrt{d_k}$$
        2. Data-Dependent Gating & Learning Rate:
           $$\beta_t = \sigma(W_\beta x_t + b_\beta) \in (0, 1) \quad \text{(Retention / Decay Gate)}$$
           $$\alpha_t = \sigma(W_\alpha x_t) \in (0, 1) \quad \text{(Delta Step Size)}$$
        3. Bidirectional Delta Rule State Recurrence:
           $$\vec{e}_t = v_t - \vec{S}_{t-1} \tilde{k}_t$$
           $$\vec{S}_t = \beta_t \vec{S}_{t-1} + \alpha_t \vec{e}_t \tilde{k}_t^\top = \beta_t \vec{S}_{t-1} + \alpha_t (v_t - \vec{S}_{t-1} \tilde{k}_t) \tilde{k}_t^\top$$
           $$\overleftarrow{e}_t = v_t - \overleftarrow{S}_{t+1} \tilde{k}_t$$
           $$\overleftarrow{S}_t = \beta_t \overleftarrow{S}_{t+1} + \alpha_t \overleftarrow{e}_t \tilde{k}_t^\top$$
        4. Bidirectional Readout & Gated Projection:
           $$\vec{y}_t = \vec{S}_t \tilde{q}_t, \quad \overleftarrow{y}_t = \overleftarrow{S}_t \tilde{q}_t$$
           $$Y = \frac{1}{2}(\vec{Y} + \overleftarrow{Y}) \odot \text{SiLU}(Z)$$
           $$\text{Output} = Y W_{\text{out}}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$ linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot d_k \cdot d_v)$ recurrent associative state.
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

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", 12)
            else:
                num_heads = kwargs.get("num_attention_heads", 12)

        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = self.d_model // self.num_heads if self.num_heads > 0 else 64

        # Linear projections
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.z_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

        # DeltaNet gating parameters
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)

        # Initialize decay bias for stable retention (~0.88)
        nn.init.constant_(self.beta_proj.bias, 2.0)
        nn.init.constant_(self.alpha_proj.bias, 0.0)

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
                self.z_proj.weight.copy_(w_q)
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    def _delta_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Sequential Delta rule associative memory scan."""
        B, H, L, d = q.shape
        S = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        step_range = range(l_idx := L - 1, -1, -1) if reverse else range(L)
        for t in step_range:
            q_t = q[:, :, t, :]  # [B, H, d]
            k_t = k[:, :, t, :]  # [B, H, d]
            v_t = v[:, :, t, :]  # [B, H, d]
            beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)    # [B, H, 1, 1]
            alpha_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            # Current prediction from prior state: Sk = S @ k_t
            # S: [B, H, d, d], k_t: [B, H, d, 1] -> [B, H, d]
            Sk = (S @ k_t.unsqueeze(-1)).squeeze(-1)
            error_t = v_t - Sk

            # Update state: S = beta * S + alpha * (error_t @ k_t^T)
            S_next = beta_t * S + alpha_t * (error_t.unsqueeze(-1) @ k_t.unsqueeze(-2))

            if mask is not None:
                m_t = mask[:, t].view(B, 1, 1, 1)
                S = torch.where(m_t, S_next, S)
            else:
                S = S_next

            # Readout: y_t = S @ q_t
            y_t = (S @ q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=2)  # [B, H, L, d]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Gated DeltaNet."""
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # 1. Projections
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        z = self.z_proj(hidden_states)

        # 2. Normalization
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1) / (d ** 0.5)

        # 3. Gating parameters: [B, H, L]
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2)
        alpha = torch.sigmoid(self.alpha_proj(hidden_states)).transpose(1, 2)

        # 4. Mask processing
        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        # 5. Bidirectional Delta Rule Scan
        y_fwd = self._delta_scan(q, k, v, beta, alpha, mask=mask_bool, reverse=False)
        y_bwd = self._delta_scan(q, k, v, beta, alpha, mask=mask_bool, reverse=True)
        y = 0.5 * (y_fwd + y_bwd)  # [B, H, L, d]

        # 6. Recombine heads and apply SiLU gating
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(y * F.silu(z))

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("gated_delta_net_inject")
def inject_gated_delta_net(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> GatedDeltaNet:
    r"""Injects GatedDeltaNet into a transformer layer."""
    d_model = 768
    num_heads = 12
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        num_heads = getattr(model_config, "num_attention_heads", 12)
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", 768)
        num_heads = getattr(layer.config, "num_attention_heads", 12)

    new_block = GatedDeltaNet(config=model_config, d_model=d_model, num_heads=num_heads, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
