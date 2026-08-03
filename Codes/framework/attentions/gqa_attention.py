"""Grouped-Query Attention (GQA).

Paper: https://arxiv.org/abs/2305.13245
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("gqa")
class GroupedQueryAttention(nn.Module):
    r"""Grouped-Query Attention (GQA) Mechanism.

    Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by dividing
    $H_Q$ query heads into $G$ groups, where each group shares a single Key-Value head pair ($H_{KV} = G \le H_Q$):
    $$K_{\text{shared}} = \text{repeat\_interleave}(K, H_Q / H_{KV}, \text{dim}=\text{heads})$$
    $$\text{Attention}_{\text{GQA}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{shared}}^\top}{\sqrt{d_k}} + M\right) V_{\text{shared}}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H_Q \cdot N^2 \cdot d_k)$
        - KV Cache Memory Reduction: Factor of $H_Q / H_{KV} = H / G$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(self, config: Any, num_kv_heads: int = 4) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.num_kv_heads: int = getattr(config, "num_kv_heads", num_kv_heads)
        self.head_dim: int = config.hidden_size // self.num_heads
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
        dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1)))
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for Grouped-Query Attention.

        Args:
            hidden_states (torch.Tensor): Input activation tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Attention mask tensor `[Batch, SeqLen]` or additive mask.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation `[Batch, SeqLen, HiddenDim]` and `None`.
        """
        B, L, _ = hidden_states.shape
        q: torch.Tensor = self.q_proj(hidden_states).view(
            B, L, self.num_heads, self.head_dim
        )
        k: torch.Tensor = self.k_proj(hidden_states).view(
            B, L, self.num_kv_heads, self.head_dim
        )
        v: torch.Tensor = self.v_proj(hidden_states).view(
            B, L, self.num_kv_heads, self.head_dim
        )

        repeat: int = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores: torch.Tensor = (
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out: torch.Tensor = (
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        )
        return self.out_proj(out), None
