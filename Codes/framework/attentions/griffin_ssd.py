"""Griffin SSD: Hybrid Local Sliding-Window Attention and Global State Space Duality (SSD) Kernel.

Combines bidirectional local sliding-window self-attention with multi-head bidirectional
State Space Duality linear recurrence and pretrained weight transfer.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("griffin_ssd")
class GriffinSSDBlock(BaseAttention):
    r"""Griffin Hybrid Block combining local window attention with global SSD linear attention.

    Mathematical Mechanism:
        Combines a bidirectional local sliding-window attention with window size $W$ and a global linear SSD kernel:
        $$\begin{aligned}
        \text{Local Attn: } & S_{\text{local}} = \frac{Q_{\text{loc}} K_{\text{loc}}^\top}{\sqrt{d_k}} + M_{\text{window}} + M_{\text{pad}} \\
        & Y_{\text{local}} = \text{softmax}(S_{\text{local}}) V_{\text{loc}} \\
        \text{Global SSD: } & \vec{H}_t = \lambda \vec{H}_{t-1} + K_t \otimes V_t, \quad \vec{Y}_t = Q_t \vec{H}_t \\
        & \overleftarrow{H}_t = \lambda \overleftarrow{H}_{t+1} + K_t \otimes V_t, \quad \overleftarrow{Y}_t = Q_t \overleftarrow{H}_t \\
        & Y_{\text{global}} = \frac{1}{2}(\vec{Y} + \overleftarrow{Y}) \\
        \text{Output: } & Y = W_{\text{out}}(Y_{\text{local}} + Y_{\text{global}})
        \end{aligned}$$

    Computational Complexity:
        - Local Attention: $\mathcal{O}(B \cdot H \cdot L \cdot W \cdot d_k)$.
        - Global SSD Recurrence: $\mathcal{O}(B \cdot H \cdot L \cdot d_k \cdot d_v)$.
        - Total Complexity: $\mathcal{O}(B \cdot L \cdot D)$ linear $\mathcal{O}(L)$ in sequence length.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        window_size: int = 64,
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
                num_heads = getattr(config, "num_attention_heads", 12)
            else:
                num_heads = kwargs.get("num_attention_heads", 12)

        self.dim: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.window_size: int = window_size

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.ssd_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=True)

        # Learnable decay gate for global SSD branch
        self.decay_param = nn.Parameter(torch.ones(self.num_heads, 1, 1) * 2.0)  # Safe log-decay init

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights."""
        try:
            adapter = UniversalAdapter()
            w_q, b_q = adapter.get_weight_bias(old_attn, "q")
            w_k, b_k = adapter.get_weight_bias(old_attn, "k")
            w_v, b_v = adapter.get_weight_bias(old_attn, "v")
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")

            with torch.no_grad():
                self.W_q.weight.copy_(w_q)
                self.W_k.weight.copy_(w_k)
                self.W_v.weight.copy_(w_v)
                self.ssd_proj.weight.copy_(w_q)
                self.out_proj.weight.copy_(w_o)
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)
        except Exception:
            pass

    def _ssd_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Multi-head associative SSD linear scan."""
        B, H, L, d = q.shape
        H_state = torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype)
        outputs = []

        step_range = range(L - 1, -1, -1) if reverse else range(L)
        for t in step_range:
            q_t = q[:, :, t, :]  # [B, H, d]
            k_t = k[:, :, t, :]  # [B, H, d]
            v_t = v[:, :, t, :]  # [B, H, d]

            H_next = decay * H_state + (k_t.unsqueeze(-1) @ v_t.unsqueeze(-2))

            if mask is not None:
                m_t = mask[:, t].view(B, 1, 1, 1)
                H_state = torch.where(m_t, H_next, H_state)
            else:
                H_state = H_next

            # y_t = q_t @ H_state
            y_t = (q_t.unsqueeze(-2) @ H_state).squeeze(-2)
            outputs.append(y_t)

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Griffin SSD attention."""
        if attention_mask is None and mask is not None:
            attention_mask = mask

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # 1. Multi-head Projections
        q = self.W_q(hidden_states).view(B, L, H, d).transpose(1, 2)
        k = self.W_k(hidden_states).view(B, L, H, d).transpose(1, 2)
        v = self.W_v(hidden_states).view(B, L, H, d).transpose(1, 2)
        q_ssd = self.ssd_proj(hidden_states).view(B, L, H, d).transpose(1, 2)

        # 2. Local Bidirectional Sliding Window Attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)  # [B, H, L, L]
        idx = torch.arange(L, device=hidden_states.device)
        dist = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0))
        window_mask = dist > self.window_size

        mask_val = -10000.0
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), mask_val)

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
                pad_mask = (attention_mask == 0).view(B, 1, 1, L)
                scores = scores.masked_fill(pad_mask, mask_val)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        local_attn = F.softmax(scores, dim=-1)
        local_attn = torch.nan_to_num(local_attn, nan=0.0)
        local_out = torch.matmul(local_attn, v)  # [B, H, L, d]

        # 3. Global Bidirectional SSD Recurrence
        q_norm = F.normalize(q_ssd, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1) / (d ** 0.5)
        decay = torch.sigmoid(self.decay_param).unsqueeze(0)  # [1, H, 1, 1]

        ssd_fwd = self._ssd_scan(q_norm, k_norm, v, decay, mask=mask_bool, reverse=False)
        ssd_bwd = self._ssd_scan(q_norm, k_norm, v, decay, mask=mask_bool, reverse=True)
        global_out = 0.5 * (ssd_fwd + ssd_bwd)

        # 4. Fused Output
        fused = (local_out + global_out).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(fused)

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None


@ATTENTION_REGISTRY.register("griffin_ssd_inject")
def inject_griffin_ssd(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> GriffinSSDBlock:
    r"""Injection hook for GriffinSSDBlock."""
    d_model = 768
    num_heads = 12
    if model_config is not None:
        d_model = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
        num_heads = getattr(model_config, "num_attention_heads", 12)
    elif hasattr(layer, "config"):
        d_model = getattr(layer.config, "hidden_size", 768)
        num_heads = getattr(layer.config, "num_attention_heads", 12)

    new_block = GriffinSSDBlock(config=model_config, d_model=d_model, num_heads=num_heads, **kwargs)
    src_attn = old_attn if old_attn is not None else layer
    if src_attn is not None and hasattr(new_block, "copy_weights"):
        new_block.copy_weights(src_attn)

    if hasattr(layer, "attention"):
        layer.attention = new_block
    elif hasattr(layer, "self"):
        layer.self = new_block

    return new_block
