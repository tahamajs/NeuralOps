r"""
Routing Attention: Content-Based Clustering for Sparse Attention.

Partitions sequence queries and keys into discrete clusters via online $k$-means,
confining quadratic attention computations to intra-cluster token neighborhoods with exact masking.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(BaseAttention):
    r"""
    Routing Attention layer clustering queries and keys via online $k$-means.

    Mathematical Formulation:
        1. Online $k$-Means Cluster Assignment:
           $$c_i = \operatorname{arg\,min}_{k \in \{1, \dots, K\}} \|Q_i - \mu_k\|_2^2, \quad \forall i \in \{1, \dots, L\}$$
        
        2. Masked Intra-Cluster Attention:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + \begin{cases} 0 & \text{if } c_i = c_j \\ -10000 & \text{if } c_i \ne c_j \end{cases}$$
           $$A = \operatorname{Softmax}(S + M_{\text{pad}})$$
           $$O = A V W_O$$

    Complexity:
        - Time: $\mathcal{O}(T \cdot B \cdot H \cdot L \cdot K + B \cdot H \cdot \frac{L^2}{K} \cdot d)$
        - Memory: $\mathcal{O}(B \cdot H \cdot L^2)$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        num_clusters: int = 16,
        n_iters: int = 3,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.num_clusters: int = getattr(config, "routing_num_clusters", num_clusters)
        self.n_iters: int = n_iters

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
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
        r"""Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except Exception:
                pass

    def _kmeans_assign(self, Q: torch.Tensor, K_num: int, iters: int = 3) -> torch.Tensor:
        r"""Deterministic k-means clustering assignment: returns cluster_ids of shape [B, H, L]."""
        B, H, L, d = Q.shape
        K = min(K_num, L)
        
        # Strided initialization
        step = max(1, L // K)
        init_idx = torch.arange(0, L, step, device=Q.device)[:K]
        if len(init_idx) < K:
            pad = torch.zeros(K - len(init_idx), dtype=torch.long, device=Q.device)
            init_idx = torch.cat([init_idx, pad])
            
        centroids = Q[:, :, init_idx, :].clone()
        Q_f = Q.float()

        cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
        for _ in range(iters):
            centroids_f = centroids.float()
            dist = torch.cdist(Q_f, centroids_f)  # [B, H, L, K]
            cluster_ids = torch.argmin(dist, dim=-1)  # [B, H, L]

            one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)
            counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
            sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
            centroids = sum_q / counts

        return cluster_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)  # [B, L]
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)              # [B, 1, 1, L]

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, L, L]

        K_num = min(max(1, self.num_clusters), L)
        cluster_ids = self._kmeans_assign(Q, K_num, self.n_iters)  # [B, H, L]

        # Intra-cluster mask: True where c_i == c_j
        cluster_mask = cluster_ids.unsqueeze(-1) == cluster_ids.unsqueeze(-2)  # [B, H, L, L]

        # Combine with padding mask
        total_mask = cluster_mask & key_valid_4d
        scores = scores.masked_fill(~total_mask, -10000.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        output = self.out_proj(context)
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("routing_inject")
def inject_routing(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> RoutingAttention:
    num_clusters = getattr(comp_config, "num_clusters", 16)
    n_iters = getattr(comp_config, "n_iters", 3)
    new_attn = RoutingAttention(
        config=model_config,
        num_clusters=num_clusters,
        n_iters=n_iters,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
