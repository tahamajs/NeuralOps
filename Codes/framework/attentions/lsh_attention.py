"""
Locality‑Sensitive Hashing (LSH) Attention – Reformer style.
Paper: https://arxiv.org/abs/2001.04451
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(nn.Module):
    r"""Locality-Sensitive Hashing (LSH) Attention (Reformer, Kitaev et al., 2020).

    Approximates dense attention by clustering keys and queries into discrete hash buckets
    via angular random hyperplane projections:
    $$h(x) = \arg\max \left( [x R; -x R] \right) \in \{0, \dots, 2 b - 1\}$$
    where $R \in \mathbb{R}^{d_k \times b}$ is a random Gaussian matrix. Tokens are sorted by hash codes
    and restricted to intra-bucket attention across $n_r$ hash rounds:
    $$\text{Attention}_{\text{LSH}}(Q, K, V) = \frac{1}{n_r} \sum_{r=1}^{n_r} \text{Softmax}\left(\frac{Q_{\mathcal{B}_k}^{(r)} (K_{\mathcal{B}_k}^{(r)})^\top}{\sqrt{d_k}} + M\right) V_{\mathcal{B}_k}^{(r)}$$

    Computational Complexity:
        - Time: $\mathcal{O}(n_r \cdot B \cdot H \cdot N \log N \cdot d_k)$
        - Memory: $\mathcal{O}(n_r \cdot B \cdot H \cdot N)$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rounds: int = 4,
        num_buckets: int = 16,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rounds: int = getattr(config, "lsh_rounds", rounds)
        self.num_buckets: int = getattr(config, "lsh_buckets", num_buckets)

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)

        # Aliases
        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy weights and biases from pretrained multi-head attention module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except Exception:
                pass

    def _hash(self, vectors: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        """Angular LSH: argmax of [x R; -x R]."""
        # vectors: (B, H, N, d), proj: (d, num_buckets // 2)
        norm = F.normalize(vectors, p=2, dim=-1)
        dots = torch.matmul(norm, proj)  # (B, H, N, b/2)
        rotations = torch.cat([dots, -dots], dim=-1)  # (B, H, N, b)
        return torch.argmax(rotations, dim=-1)  # (B, H, N) with values in {0, ..., b-1}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for multi-round locality-sensitive hashing attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        K: torch.Tensor = self.key(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        V: torch.Tensor = self.value(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        scale: float = d**-0.5

        out: torch.Tensor = torch.zeros_like(V)
        half_buckets = max(1, self.num_buckets // 2)

        for r in range(self.rounds):
            # Random projection matrix for this round
            proj = torch.randn(d, half_buckets, device=device, dtype=dtype)
            proj = F.normalize(proj, p=2, dim=0)

            q_hash = self._hash(Q, proj)  # (B, H, N)
            k_hash = self._hash(K, proj)  # (B, H, N)

            # Bucket-wise attention computation
            for b in range(B):
                v_m = valid_mask[b]
                for h in range(H):
                    qh = q_hash[b, h]
                    kh = k_hash[b, h]
                    for bucket_id in torch.unique(qh):
                        q_idx = (qh == bucket_id) & v_m
                        k_idx = (kh == bucket_id) & v_m

                        if not q_idx.any() or not k_idx.any():
                            continue

                        q_b = Q[b, h, q_idx]  # (n_q, d)
                        k_b = K[b, h, k_idx]  # (n_k, d)
                        v_b = V[b, h, k_idx]  # (n_k, d)

                        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale  # (n_q, n_k)
                        attn = F.softmax(scores, dim=-1)
                        attn = self.dropout(attn)
                        ctx = torch.matmul(attn, v_b)  # (n_q, d)
                        out[b, h, q_idx] += ctx

        out = out / max(1, self.rounds)
        if head_mask is not None:
            out = out * head_mask

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(out)
        out = out * valid_mask.unsqueeze(-1).to(dtype)
        return out, None
