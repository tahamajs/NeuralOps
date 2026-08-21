r"""
Weighted Grouped-Query Attention (WGQA).

Extends standard Grouped-Query Attention (GQA) with learnable per-channel
importance weights applied independently to each KV head prior to query-head expansion.

Mathematical Mechanism:
    Given $H_Q$ query heads and $H_{KV}$ KV heads ($H_{KV} \leq H_Q$),
    with learnable scalar weight vectors $w_K, w_V \in \mathbb{R}^{H_{KV}}$:
    1. **Weighted Key-Value Scaling:**
       $$\tilde{K}_h = w_{K, h} \cdot K_h, \quad \tilde{V}_h = w_{V, h} \cdot V_h$$
    2. **Repeat-Interleave Expansion:**
       $$K_{\text{exp}} = \text{repeat\_interleave}(\tilde{K}, H_Q / H_{KV}, \text{dim=heads})$$
       $$V_{\text{exp}} = \text{repeat\_interleave}(\tilde{V}, H_Q / H_{KV}, \text{dim=heads})$$
    3. **Scaled Dot-Product Attention:**
       $$A = \text{Softmax}\left(\frac{Q K_{\text{exp}}^\top}{\sqrt{d_k}} + M\right) V_{\text{exp}}$$
       $$\text{Output} = W_O\, \text{Concat}(A_1, \dots, A_{H_Q})$$

Computational Complexity:
    - Time: $\mathcal{O}(B \cdot H_Q \cdot L^2 \cdot d_k)$ — same as GQA.
    - KV Cache Memory: $\mathcal{O}(B \cdot H_{KV} \cdot L \cdot d_k)$, reduced by $H_Q / H_{KV}$.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("wgqa")
class WeightedGroupedQueryAttention(nn.Module):
    r"""Weighted Grouped-Query Attention with learnable per-channel KV importance weights.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        num_kv_heads: Number of compressed Key-Value heads ($H_{KV}$).
    """

    def __init__(self, config: Any, num_kv_heads: int = 4) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = config.hidden_size // self.num_heads

        # Learnable per-KV-head importance weights
        self.k_weights: nn.Parameter = nn.Parameter(
            torch.ones(self.num_kv_heads, 1)
        )  # (H_kv, 1)
        self.v_weights: nn.Parameter = nn.Parameter(torch.ones(self.num_kv_heads, 1))

        self.q_proj: nn.Linear = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim
        )
        self.k_proj: nn.Linear = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.v_proj: nn.Linear = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.out_proj: nn.Linear = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size
        )
        dropout_p: float = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute weighted grouped-query attention.

        Args:
            hidden_states: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or additive `[Batch, 1, SeqLen, SeqLen]`.
            head_mask: Optional head pruning mask.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected representation `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        # Apply learnable per-head importance scaling
        k = k * self.k_weights.view(1, 1, self.num_kv_heads, 1)
        v = v * self.v_weights.view(1, 1, self.num_kv_heads, 1)

        # Repeat-interleave to match query heads
        repeat_factor: int = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.transpose(1, 2)  # (B, H_q, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Usually (B, L) with 0 for pad and 1 for valid
                scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask
            else:
                scores = scores + attention_mask.unsqueeze(1).unsqueeze(2)

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(context)

        if output_attentions:
            return out, attn_probs
        return out, None
