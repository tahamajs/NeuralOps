r"""
Quality and Capacity-Aware Grouped Query Attention (QCQA).

Dynamically synthesizes head-specific key-value representations for each query head
via continuous learned routing distributions over a compressed set of KV heads.

Mathematical Mechanism:
    Given $H_q$ query heads and a reduced set of $H_{kv}$ key-value heads ($H_{kv} \le H_q$):
    1. **Dynamic Soft Routing Distribution:**
       For each token position and query head $h \in \{1, \dots, H_q\}$:
       $$\pi_{h, j} = \frac{\exp\left(\frac{q_h W_{\text{router}, j}}{\tau}\right)}{\sum_{j'=1}^{H_{kv}} \exp\left(\frac{q_h W_{\text{router}, j'}}{\tau}\right)}, \quad \forall j \in \{1, \dots, H_{kv}\}$$
    2. **Head Mixture Synthesis:**
       $$\widetilde{K}_h = \sum_{j=1}^{H_{kv}} \pi_{h, j} K_j \in \mathbb{R}^{B \times L \times d}, \quad \widetilde{V}_h = \sum_{j=1}^{H_{kv}} \pi_{h, j} V_j \in \mathbb{R}^{B \times L \times d}$$
    3. **Standard Attention Computation with Synthesized KV:**
       $$A_h = \operatorname{Softmax}\left(\frac{Q_h \widetilde{K}_h^T}{\sqrt{d}} + M\right) \in \mathbb{R}^{B \times L \times L}$$
       $$O_h = A_h \widetilde{V}_h \in \mathbb{R}^{B \times L \times d}$$

Computational Complexity:
    - Routing projection & mixture: $\mathcal{O}(B \cdot L \cdot H_q \cdot H_{kv} \cdot d)$.
    - Attention matrix products: $\mathcal{O}(B \cdot H_q \cdot L^2 \cdot d)$.
    - KV Cache footprint during autoregressive decoding is reduced by factor $H_q / H_{kv}$.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("qcqa")
class QCQAAttention(nn.Module):
    r"""Quality and Capacity-Aware Grouped Query Attention layer.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        num_kv_heads: Number of compressed key-value heads ($H_{kv}$).
        router_temp: Temperature scaling factor $\tau$ for soft routing softmax.
    """

    def __init__(
        self,
        config: Any,
        num_kv_heads: int = 4,
        router_temp: float = 1.0,
    **kwargs) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)

        self.router = nn.Linear(self.head_dim, self.num_kv_heads, bias=False)
        self.router_temp: float = router_temp

        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute QCQA dynamic mixture attention.

        Args:
            hidden_states: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor `[Batch, SeqLen]` or broadcastable.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected attention output `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax attention probability tensor `[Batch, NumHeads, SeqLen, SeqLen]`.
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

        route_logits = self.router(q) / self.router_temp
        route_weights = F.softmax(route_logits, dim=-1)  # (B, L, H_q, H_kv)

        B, L, H_q, H_kv = route_weights.shape
        k_flat = k.reshape(B * L, H_kv, self.head_dim)
        w_flat = route_weights.reshape(B * L, H_q, H_kv)
        k_mixed_flat = torch.bmm(w_flat, k_flat)
        k_mixed = k_mixed_flat.reshape(B, L, H_q, self.head_dim)

        v_flat = v.reshape(B * L, H_kv, self.head_dim)
        v_mixed_flat = torch.bmm(w_flat, v_flat)
        v_mixed = v_mixed_flat.reshape(B, L, H_q, self.head_dim)

        q = q.transpose(1, 2)  # (B, H_q, L, D)
        k_mixed = k_mixed.transpose(1, 2)
        v_mixed = v_mixed.transpose(1, 2)

        scores = torch.matmul(q, k_mixed.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1)
            else:
                scores = scores + attention_mask

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v_mixed)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(context)

        return out, attn_probs
