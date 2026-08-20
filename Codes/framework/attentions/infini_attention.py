import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("infini")
class InfiniAttention(BaseAttention):
    r"""
    Infini-Attention with Compressive Memory and Sliding Window.

    Mathematical Mechanism:
        Combines masked local sliding-window attention of size $W$ with a compact compressive memory
        tensor $M \in \mathbb{R}^{1 \times H \times C \times d_k}$:
        $$\begin{aligned}
        Q &= X W_Q, \quad K = X W_K, \quad V = X W_V \\
        A_{\text{mem}} &= \text{softmax}\left(\frac{Q M^\top}{\sqrt{d_k}}\right) M \\
        S_{\text{local}} &= \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{window}} \\
        A_{\text{local}} &= \text{softmax}(S_{\text{local}}) V \\
        O &= (A_{\text{local}} + A_{\text{mem}}) W_O
        \end{aligned}$$

    Computational Complexity:
        - Sliding-Window Attention: $\mathcal{O}(B \cdot H \cdot L \cdot W \cdot d_k)$
        - Compressive Memory: $\mathcal{O}(B \cdot H \cdot L \cdot C \cdot d_k)$
        - Total Time: $\mathcal{O}(B \cdot H \cdot L \cdot (W + C) \cdot d_k)$ — strictly linear in sequence length $L$.
        - Space: $\mathcal{O}(B \cdot H \cdot L \cdot W)$ for masked attention weights.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        compress_dim: int = 128,
        local_window: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.compress_dim: int = compress_dim
        self.local_window: int = local_window
        dropout_p: float = getattr(config, "dropout", 0.0)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)

        # Compressive memory matrix (learnable)
        self.memory: nn.Parameter = nn.Parameter(
            torch.randn(1, self.num_heads, compress_dim, self.head_dim) * 0.02
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of InfiniAttention.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor of shape `[B, L]` or `[B, 1, L, L]`.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return local attention matrix.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_local)` where `output` is `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        # Compressive memory attention
        mem = self.memory.expand(B, -1, -1, -1)  # (B, H, C, Dh)
        mem_scores = torch.matmul(q, mem.transpose(-2, -1)) / math.sqrt(Dh)
        mem_attn = torch.softmax(mem_scores, dim=-1)
        mem_out = torch.matmul(mem_attn, mem)

        # Sliding-window attention
        idx = torch.arange(L, device=q.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.local_window
        window_mask = causal & window

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow]
        mask_val = -10000.0 if scores.dtype in (torch.float16, torch.bfloat16) else -10000.0
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), mask_val)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2) == 0
                scores = scores.masked_fill(pad_mask, mask_val)
            else:
                scores = scores + (attention_mask.to(dtype=scores.dtype) - 1.0) * abs(mask_val)

        attn_local = F.softmax(scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.matmul(attn_local, v)

        out = local_out + mem_out
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out, (attn_local if output_attentions else None)
