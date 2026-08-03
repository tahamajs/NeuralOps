r"""
Native Sparse Attention (NSA) Module.

Combines sliding-window local context capture with dynamic, input-dependent top-$k$
global routing across key-value representations.

Mathematical Mechanism:
    For queries $Q \in \mathbb{R}^{B \times H \times L \times d}$ and keys $K \in \mathbb{R}^{B \times H \times L \times d}$:
    $$\text{Scores}_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}$$
    The attention pattern is constrained by a union of a local band mask $\mathcal{M}_{\text{local}}$ and top-$k$ routing:
    $$\mathcal{M}_{ij} = \begin{cases}
    1 & \text{if } |i - j| \le W \\
    1 & \text{if } j \in \operatorname{top-}k(\{ \text{Scores}_{ij'} : |i - j'| > W \}) \\
    0 & \text{otherwise}
    \end{cases}$$
    $$\text{Attention}(Q, K, V) = \operatorname{Softmax}(\text{Scores} + (1 - \mathcal{M}) \cdot (-\infty)) V$$

Computational Complexity:
    - Sparse evaluation scales with $\mathcal{O}(B \cdot H \cdot L \cdot (2W + k) \cdot d)$.
    - Dense dot-product fallback computation is $\mathcal{O}(B \cdot H \cdot L^2 \cdot d)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nsa")
class NativeSparseAttention(nn.Module):
    r"""Native Sparse Attention layer combining local windowing and global top-k selection.

    Args:
        config: Model configuration specifying `num_attention_heads`, `hidden_size`, and optional `dropout`.
        window_size: Half-width of sliding local window ($W$).
        top_k: Number of highest-scoring non-local tokens to retain per query ($k$).
    """

    def __init__(self, config: Any, window_size: int = 128, top_k: int = 32) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.window_size: int = window_size
        self.top_k: int = top_k
        dropout_p: float = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute native sparse attention.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor of shape `[Batch, SeqLen]` or broadcastable.
            **kwargs: Additional layer keyword arguments.

        Returns:
            Tuple of:
                - output: Attended and projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Optional attention matrix or None.
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        scale = d**-0.5
        idx = torch.arange(L, device=Q.device)
        local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        topk_mask = torch.zeros_like(scores)

        for i in range(L):
            s_i = scores[:, :, i, :]
            s_i = s_i.masked_fill(
                local_mask[0, 0, i].unsqueeze(0).unsqueeze(0), float("-inf")
            )
            k_val = min(self.top_k, L)
            _, topk_ind = torch.topk(s_i, k_val, dim=-1)
            topk_mask[:, :, i, :].scatter_(-1, topk_ind, 1.0)

        total_mask = (topk_mask + local_mask.float()) > 0
        scores = scores.masked_fill(~total_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask.to(scores.dtype)) * -1e9

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        output = self.out_proj(context)
        return (output, None)
