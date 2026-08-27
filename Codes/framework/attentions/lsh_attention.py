r"""
Locality-Sensitive Hashing (LSH) Attention – Reformer style.
Paper: https://arxiv.org/abs/2001.04451

Approximates dense attention by clustering keys and queries into discrete hash buckets
via angular random hyperplane projections with fallback context and exact pretrained weight transfer.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("lsh")
class LSHAttention(BaseAttention):
    r"""
    Locality-Sensitive Hashing (LSH) Attention (Reformer, Kitaev et al., 2020).

    Mathematical Formulation:
        1. Angular LSH Hashing:
           $$h(x) = \arg\max \left( [x R; -x R] \right) \in \{0, \dots, 2b - 1\}$$
           where $R \in \mathbb{R}^{d_k \times b}$ is a normalized Gaussian random rotation matrix.
        
        2. Multi-Round Hashing & Bucket Attention:
           $$\text{Attention}_{\text{LSH}}(Q, K, V) = \frac{1}{n_r} \sum_{r=1}^{n_r} \operatorname{Softmax}\left(\frac{Q_{\mathcal{B}_k}^{(r)} (K_{\mathcal{B}_k}^{(r)})^\top}{\sqrt{d_k}} + M\right) V_{\mathcal{B}_k}^{(r)}$$

    Complexity:
        - Time: $\mathcal{O}(n_r \cdot B \cdot H \cdot N \log N \cdot d_k)$
        - Memory: $\mathcal{O}(n_r \cdot B \cdot H \cdot N)$ vs $\mathcal{O}(B \cdot H \cdot N^2)$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        rounds: int = 4,
        num_buckets: int = 16,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 4)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rounds: int = getattr(config, "lsh_rounds", rounds)
        self.num_buckets: int = getattr(config, "lsh_buckets", num_buckets)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Copy weights and biases from pretrained multi-head attention module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def _hash(self, vectors: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        r"""Angular LSH: argmax of [x R; -x R]. vectors: (B, H, N, d), proj: (d, b/2)"""
        norm = F.normalize(vectors, p=2, dim=-1)
        dots = torch.matmul(norm, proj)  # (B, H, N, b/2)
        rotations = torch.cat([dots, -dots], dim=-1)  # (B, H, N, b)
        return torch.argmax(rotations, dim=-1)  # (B, H, N) in {0, ..., b-1}

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
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        scale = d ** -0.5

        out = torch.zeros_like(V)
        half_buckets = max(1, self.num_buckets // 2)

        for r in range(self.rounds):
            # Seeded deterministic random projection per round
            generator = torch.Generator(device=device)
            generator.manual_seed(r * 1000 + N)
            proj = torch.randn(d, half_buckets, device=device, dtype=dtype, generator=generator)
            proj = F.normalize(proj, p=2, dim=0)

            q_hash = self._hash(Q, proj)  # (B, H, N)
            k_hash = self._hash(K, proj)  # (B, H, N)

            # Vectorized bucket-wise mask: tokens in the same hash bucket attend to each other
            bucket_mask = (q_hash.unsqueeze(-1) == k_hash.unsqueeze(-2))  # (B, H, N, N)
            valid_2d = valid_mask.unsqueeze(1).unsqueeze(2) & valid_mask.unsqueeze(1).unsqueeze(3)  # (B, 1, N, N)
            combined_mask = bucket_mask & valid_2d

            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)
            scores = scores.masked_fill(~combined_mask, -10000.0)

            attn = F.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, 0.0)
            attn = self.dropout(attn)
            ctx = torch.matmul(attn, V)  # (B, H, N, d)
            out = out + ctx

        out = out / max(1, self.rounds)
        if head_mask is not None:
            out = out * head_mask

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(out)
        out = out * valid_mask.unsqueeze(-1).to(dtype)
        return out, None


@ATTENTION_REGISTRY.register("lsh_inject")
def inject_lsh(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs) -> LSHAttention:
    rounds = getattr(comp_config, "lsh_rounds", 4)
    buckets = getattr(comp_config, "lsh_buckets", 16)
    new_attn = LSHAttention(
        config=model_config,
        rounds=rounds,
        num_buckets=buckets,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
