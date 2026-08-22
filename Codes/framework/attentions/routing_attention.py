r"""
Routing Attention: Content-Based Clustering for Sparse Attention.

Partitions sequence queries and keys into discrete clusters via online $k$-means,
confining quadratic attention computations to intra-cluster token neighborhoods.

Mathematical Mechanism:
    Given queries $Q \in \mathbb{R}^{B \times H \times L \times d}$ and keys $K \in \mathbb{R}^{B \times H \times L \times d}$:
    1. **Iterative Head-Wise $k$-Means Clustering:**
       Initialize $K$ cluster centroids $\{\mu_k^{(0)}\}_{k=1}^K$. For $t = 1, \dots, T$:
       $$c_i^{(t)} = \operatorname{arg\,min}_{k \in \{1, \dots, K\}} \|Q_i - \mu_k^{(t-1)}\|_2^2, \quad \forall i \in \{1, \dots, L\}$$
       $$\mu_k^{(t)} = \frac{\sum_{i=1}^L \mathbb{I}(c_i^{(t)} = k) Q_i}{\max\left(1, \sum_{i=1}^L \mathbb{I}(c_i^{(t)} = k)\right)}$$
    2. **Intra-Cluster Attention Evaluation:**
       For each cluster $k \in \{1, \dots, K\}$:
       $$A_{ij}^{(k)} = \operatorname{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right) \cdot \mathbb{I}(c_i = k)\mathbb{I}(c_j = k)$$
       $$O_i = \sum_{j \in \mathcal{C}_{c_i}} A_{ij} V_j$$

Computational Complexity:
    - $k$-means assignment & centroid update: $\mathcal{O}(T \cdot B \cdot H \cdot L \cdot K \cdot d)$.
    - Clustered attention: $\mathcal{O}(B \cdot H \cdot K \cdot (L / K)^2 \cdot d) = \mathcal{O}(B \cdot H \cdot \frac{L^2}{K} \cdot d)$.
    - Overall Complexity: $\mathcal{O}(T \cdot L \cdot K \cdot d + \frac{L^2}{K} \cdot d) \ll \mathcal{O}(L^2 \cdot d)$ when choosing $K \approx \sqrt{L}$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(nn.Module):
    r"""Routing Attention layer clustering queries via online $k$-means.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        num_clusters: Number of geometric cluster centroids ($K$).
        n_iters: Number of Lloyd $k$-means iterations ($T$).
        pretrained_attn: Optional source module for weight copying.
    """

    def __init__(
        self,
        config: Any,
        num_clusters: int = 32,
        n_iters: int = 3,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.num_clusters: int = num_clusters
        self.n_iters: int = n_iters
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Copy projection weights from pretrained attention module."""
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _kmeans_assign(self, Q: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        r"""Assign queries to nearest centroids via Euclidean distance."""
        # FIX: [Cast to float32 for torch.cdist execution to ensure CPU float16/bfloat16 compatibility]
        Q_f = Q.float() if Q.dtype in (torch.float16, torch.bfloat16) else Q
        centroids_f = centroids.float() if centroids.dtype in (torch.float16, torch.bfloat16) else centroids
        dist = torch.cdist(Q_f, centroids_f)  # (B, H, L, K)
        return torch.argmin(dist, dim=-1)  # (B, H, L)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Compute routing cluster attention.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head masking tensor.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention matrix or None.
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                valid = (attention_mask[:, 0, 0, :] > -10000.0)
            elif attention_mask.dim() == 3:
                valid = (attention_mask[:, 0, :] > -10000.0)
            else:
                valid = attention_mask.bool()
                
            valid_4d = valid.unsqueeze(1).unsqueeze(-1).to(Q.dtype)
            Q = Q * valid_4d
            K = K * valid_4d
            V = V * valid_4d
        else:
            valid = None

        num_clusters = min(self.num_clusters, L)
        centroids = Q[:, :, :num_clusters, :]

        for _ in range(self.n_iters):
            cluster_ids = self._kmeans_assign(Q, centroids)
            new_centroids = torch.zeros_like(centroids)
            for k in range(num_clusters):
                mask = (cluster_ids == k).unsqueeze(-1)
                sum_q = (Q * mask).sum(dim=2)
                count = mask.sum(dim=2).clamp(min=1)
                new_centroids[:, :, k, :] = sum_q / count
            centroids = new_centroids

        cluster_ids = self._kmeans_assign(Q, centroids)
        out = torch.zeros_like(V)
        scale = d**-0.5
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow]
        mask_val = -10000.0 if Q.dtype in (torch.float16, torch.bfloat16) else -10000.0

        for k in range(num_clusters):
            mask = (cluster_ids == k).unsqueeze(-1)
            if mask.sum() == 0:
                continue
            Qk = Q * mask
            Kk = K * mask
            Vk = V * mask
            scores = torch.matmul(Qk, Kk.transpose(-2, -1)) * scale
            if valid is not None:
                key_valid = valid.unsqueeze(1).unsqueeze(-2).to(scores.dtype)
                scores = scores.masked_fill(key_valid == 0, mask_val)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out += torch.matmul(attn, Vk)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out, None


def inject_routing(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> RoutingAttention:
    r"""Bridge helper injecting RoutingAttention into a target transformer layer."""
    num_clusters = getattr(comp_config, "num_clusters", 32)
    n_iters = getattr(comp_config, "n_iters", 3)
    new_attn = RoutingAttention(
        config=model_config,
        num_clusters=num_clusters,
        n_iters=n_iters,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
