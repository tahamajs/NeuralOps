import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def ssd_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""
    Causal State Space Duality (SSD) Recurrent Linear Attention Kernel.

    Mathematical Mechanism:
        Maintains an associative 1-step recurrent matrix state $H_t \in \mathbb{R}^{D \times D}$:
        $$\begin{aligned}
        H_t &= H_{t-1} + k_t \otimes v_t = H_{t-1} + k_t v_t^\top \\
        y_t &= q_t^\top H_t = \sum_{j=1}^{t} (q_t^\top k_j) v_j
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot L \cdot D^2)$
        - Space: $\mathcal{O}(B \cdot D^2)$ state memory per sequence.

    Args:
        q: Query tensor of shape `[B, L, D]`.
        k: Key tensor of shape `[B, L, D]`.
        v: Value tensor of shape `[B, L, D]`.

    Returns:
        Recurrently accumulated output tensor of shape `[B, L, D]`.
    """
    B, L, D = q.shape
    h = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    for t in range(L):
        kt = k[:, t]  # (B, D)
        vt = v[:, t]  # (B, D)
        qt = q[:, t]  # (B, D)
        h = h + kt.unsqueeze(-1) @ vt.unsqueeze(1)  # outer product (B, D, D)
        out_t = (qt.unsqueeze(1) @ h).squeeze(1)  # (B, D)
        outputs.append(out_t)
    return torch.stack(outputs, dim=1)


@ATTENTION_REGISTRY.register("griffin_ssd")
class GriffinSSDBlock(BaseAttention):
    r"""
    Griffin Hybrid Block combining local sliding-window attention with global SSD linear attention.

    Mathematical Mechanism:
        Combines a local causal sliding-window attention with window size $W$ and a global linear SSD kernel:
        $$\begin{aligned}
        \text{Local Attn: } & S_{\text{local}} = \frac{Q_{\text{loc}} K_{\text{loc}}^\top}{\sqrt{D}} + M_{\text{window}} \\
        & Y_{\text{local}} = \text{softmax}(S_{\text{local}}) V_{\text{loc}} \\
        \text{Global SSD: } & Y_{\text{global}} = \text{ssd\_kernel}(Q_{\text{glob}}, K_{\text{glob}}, V_{\text{glob}}) \\
        \text{Output: } & Y = W_{\text{out}}(Y_{\text{local}} + Y_{\text{global}})
        \end{aligned}$$

    Computational Complexity:
        - Local Attention: $\mathcal{O}(B \cdot L \cdot W \cdot D)$ time, $\mathcal{O}(B \cdot L \cdot W)$ memory.
        - Global SSD Recurrence: $\mathcal{O}(B \cdot L \cdot D^2)$ time, $\mathcal{O}(B \cdot D^2)$ state memory.
        - Overall Complexity: $\mathcal{O}(B \cdot L \cdot (W \cdot D + D^2))$ — strictly linear $\mathcal{O}(L)$ in sequence length.

    Tensor Dimensions:
        - Input $x$: $[B, L, D]$
        - Output: Tuple `(out, None)` where `out` is $[B, L, D]$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 64,
        window_size: int = 256,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", 768)
        self.window_size: int = window_size

        self.W_q: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.ssd_proj: nn.Linear = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj: nn.Linear = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for Griffin SSD attention.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional padding mask of shape `[B, L]` or `[B, 1, L, L]`.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for consistency.
            mask: Optional alias for attention mask.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, L, D]`.
        """
        if attention_mask is None and mask is not None:
            attention_mask = mask
        B, L, D = hidden_states.shape

        # Local sliding window (causal)
        q_local = self.W_q(hidden_states)
        k_local = self.W_k(hidden_states)
        v_local = self.W_v(hidden_states)
        scores = torch.bmm(q_local, k_local.transpose(1, 2)) / (math.sqrt(D))
        idx = torch.arange(L, device=hidden_states.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        local_mask = ~(causal & window)
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow]
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask.unsqueeze(1) == 0  # (B, 1, L)
                scores = scores.masked_fill(pad_mask, mask_val)
        scores = scores.masked_fill(local_mask.unsqueeze(0), mask_val)
        local_out = torch.bmm(F.softmax(scores, dim=-1), v_local)

        # Global linear attention
        q_global = self.ssd_proj(hidden_states)
        k_global = self.W_k(hidden_states)
        v_global = self.W_v(hidden_states)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                m = attention_mask.unsqueeze(-1).to(dtype=k_global.dtype)
                k_global = k_global * m
                v_global = v_global * m
        global_out = ssd_kernel(q_global, k_global, v_global)

        out = self.out_proj(local_out + global_out)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                out = out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, None
