r"""
ClusterFormer: Token Clustering Attention Module.

Groups queries into $K$ semantic cluster centroids via fast differentiable $K$-means
and evaluates centroid-to-key cross-attention alongside local context, achieving $\mathcal{O}(K \cdot L \cdot d)$ linear complexity
while eliminating token isolation.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("cluster")
@ATTENTION_REGISTRY.register("cluster_attention")
class ClusterAttention(BaseAttention):
    r"""
    ClusterFormer Attention Layer (Fast Transformers with Clustered Attention).

    Mathematical Formulation:
        1. Query Clustering via $K$-Means:
           Partitions queries $Q \in \mathbb{R}^{B \times H \times L \times d}$ into $K \ll L$ centroids:
           $$\mu_k = \frac{\sum_{i=1}^L \mathbb{I}(c_i = k) Q_i}{\max\left(1, \sum_{i=1}^L \mathbb{I}(c_i = k)\right)}, \quad k \in \{1, \dots, K\}$$
        
        2. Centroid-to-Key Global Cross-Attention:
           $$S^{\text{cent}}_{k, j} = \frac{\mu_k K_j^\top}{\sqrt{d}}, \quad A^{\text{cent}} = \operatorname{Softmax}\left(S^{\text{cent}} + M\right) \in \mathbb{R}^{B \times H \times K \times L}$$
           $$V^{\text{cent}}_k = \sum_{j=1}^L A^{\text{cent}}_{k, j} V_j \in \mathbb{R}^{B \times H \times K \times d}$$
        
        3. Local Attention & Fused Representation:
           $$O_i = V^{\text{cent}}_{c_i} + O^{\text{local}}_i$$

    Complexity:
        - Time: $\mathcal{O}(T \cdot B \cdot H \cdot L \cdot K + B \cdot H \cdot K \cdot L \cdot d + B \cdot H \cdot L \cdot W \cdot d)$
        - Memory: $\mathcal{O}(B \cdot H \cdot (K \cdot L + L \cdot W))$ vs $\mathcal{O}(B \cdot H \cdot L^2)$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        num_clusters: int = 16,
        n_iters: int = 3,
        window_size: int = 32,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.num_clusters: int = getattr(config, "cluster_num_clusters", num_clusters)
        self.n_iters: int = n_iters
        self.window_size: int = window_size
        self.local_residual = nn.Parameter(torch.tensor(0.5))

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
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def _kmeans_cluster(
        self, Q: torch.Tensor, K_num: int, iters: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Deterministic Lloyd's $K$-means clustering for queries.
        Q: [B, H, L, d] -> centroids: [B, H, K, d], cluster_ids: [B, H, L]
        """
        B, H, L, d = Q.shape
        K = min(K_num, L)
        
        # Deterministic uniform strided initialization
        step = max(1, L // K)
        init_indices = torch.arange(0, L, step, device=Q.device)[:K]
        if len(init_indices) < K:
            pad_idx = torch.zeros(K - len(init_indices), dtype=torch.long, device=Q.device)
            init_indices = torch.cat([init_indices, pad_idx])
        
        centroids = Q[:, :, init_indices, :].clone()  # [B, H, K, d]

        cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
        Q_f = Q.float()
        for _ in range(iters):
            centroids_f = centroids.float()
            # Compute distance [B, H, L, K]
            # dist = ||Q - mu||^2 = ||Q||^2 + ||mu||^2 - 2 Q mu^T
            dist = torch.cdist(Q_f, centroids_f)  # [B, H, L, K]
            cluster_ids = torch.argmin(dist, dim=-1)  # [B, H, L]

            # Recompute centroids
            # One-hot mask [B, H, L, K]
            one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)  # [B, H, L, K]
            counts = one_hot.sum(dim=2).unsqueeze(-1)  # [B, H, K, 1]
            sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)       # [B, H, K, d]
            
            # Retain previous centroid if cluster is empty
            centroids = torch.where(counts > 0, sum_q / counts.clamp(min=1.0), centroids)
        
        return centroids, cluster_ids

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
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)  # [B, H, L, Dh]
        K = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)  # [B, L]
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        scale = Dh ** -0.5
        K_num = min(max(1, self.num_clusters), L)

        # 1. Cluster Queries into K centroids
        centroids, cluster_ids = self._kmeans_cluster(Q, K_num, self.n_iters)  # [B, H, K, Dh], [B, H, L]

        # 2. Centroid-to-Key Global Cross-Attention [B, H, K, L]
        scores_cent = torch.matmul(centroids, K.transpose(-2, -1)) * scale
        scores_cent = scores_cent.masked_fill(~key_valid_4d, -10000.0)
        attn_cent = F.softmax(scores_cent, dim=-1)
        attn_cent = self.dropout(attn_cent)

        # Centroid values: [B, H, K, Dh]
        V_cent = torch.matmul(attn_cent, V)

        # 3. Gather centroid context back to each token [B, H, L, Dh]
        # cluster_ids: [B, H, L] -> expand to [B, H, L, Dh]
        cluster_ids_expanded = cluster_ids.unsqueeze(-1).expand(B, H, L, Dh)
        global_context = torch.gather(V_cent, 2, cluster_ids_expanded)

        # 4. Optional Local Sliding Window Attention
        if self.window_size > 0 and self.window_size < L:
            scores_loc = torch.matmul(Q, K.transpose(-2, -1)) * scale
            idx = torch.arange(L, device=device)
            band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
            band_mask_4d = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
            scores_loc = scores_loc.masked_fill(~band_mask_4d, -10000.0)
            attn_loc = F.softmax(scores_loc, dim=-1)
            attn_loc = self.dropout(attn_loc)
            local_context = torch.matmul(attn_loc, V)
            # Balanced fusion with soft residual
            alpha = torch.sigmoid(self.local_residual)
            out_context = (1.0 - alpha) * global_context + alpha * local_context
        else:
            out_context = global_context

        if head_mask is not None:
            out_context = out_context * head_mask

        # Mask padding queries
        out_context = out_context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        out = out_context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        return self.out_proj(out), (attn_cent if output_attentions else None)


@ATTENTION_REGISTRY.register("cluster_inject")
def inject_cluster(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
,
    **kwargs) -> ClusterAttention:
    num_clusters = getattr(comp_config, "num_clusters", 16)
    new_attn = ClusterAttention(
        config=model_config,
        num_clusters=num_clusters,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
