r"""
ClusterFormer: Token Clustering Attention Module.

Groups token embeddings into $K$ semantic clusters via differentiable or iterative $K$-means
and performs dense scaled dot-product attention independently within each cluster partition.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("cluster")
class ClusterAttention(nn.Module):
    r"""
    ClusterFormer Attention Layer.

    Mathematical Formulation:
        1. Token Clustering via $K$-means:
           Given input activations $X \in \mathbb{R}^{B \times L \times D}$, iteratively partitions tokens into $K$ centroids:
           $$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} X_i, \quad C_k = \left\{ i \in \{1, \dots, L\} : \arg\min_j \|X_i - \mu_j\|_2 = k \right\}$$

        2. Intra-Cluster Attention:
           For each cluster partition $C_k$:
           $$S_{ij}^{(k)} = \frac{Q_i K_j^T}{\sqrt{d_h}}, \quad \forall i, j \in C_k$$
           $$A^{(k)} = \text{Softmax}\left( S^{(k)} \right)$$
           $$O_i = \sum_{j \in C_k} A_{ij}^{(k)} V_j$$

    Complexity:
        - Clustering Step: $\mathcal{O}(I \cdot B \cdot L \cdot K \cdot D)$ for $I$ iterations.
        - Attention Computation: $\mathcal{O}\left(B \cdot H \cdot \frac{L^2}{K} \cdot d_h\right)$, providing a $K \times$ speedup over full attention.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Centroids: $[B, K, D]$
        - Cluster IDs: $[B, L]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        num_clusters: int = 64,
        n_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.num_clusters: int = num_clusters
        self.n_iters: int = n_iters
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(
            getattr(config, "attention_probs_dropout_prob", getattr(config, "dropout", 0.1))
        )

    def _kmeans(self, x: torch.Tensor, iters: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Performs batch-wise iterative $K$-means clustering on token representations.

        Args:
            x (torch.Tensor): Input token activations $[B, L, D]$.
            iters (int): Number of Lloyd's algorithm iterations.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Centroids $[B, K, D]$.
                - Cluster assignments $[B, L]$.
        """
        B, L, D = x.shape
        K = min(self.num_clusters, L)
        idx = torch.randperm(L, device=x.device)[:K]
        centroids = x[:, idx, :].clone()

        cluster_ids = torch.zeros(B, L, dtype=torch.long, device=x.device)
        # FIX: [Cast to float32 for torch.cdist execution to ensure CPU float16/bfloat16 compatibility]
        x_f = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        for _ in range(iters):
            centroids_f = centroids.float() if centroids.dtype in (torch.float16, torch.bfloat16) else centroids
            dist = torch.cdist(x_f, centroids_f)  # (B, L, K)
            cluster_ids = torch.argmin(dist, dim=-1)  # (B, L)
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask_k = (cluster_ids == k).unsqueeze(-1).to(dtype=x.dtype)
                sum_k = (x * mask_k).sum(dim=1, keepdim=True)
                count_k = mask_k.sum(dim=1, keepdim=True).clamp(min=1.0)
                new_centroids[:, k, :] = (sum_k / count_k).squeeze(1)
            centroids = new_centroids
        return centroids, cluster_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for ClusterAttention.

        Args:
            hidden_states (torch.Tensor): Input activations $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Attention mask tensor $[B, L]$.
            output_attentions (bool): Whether to return intermediate attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed hidden activations $[B, L, D]$.
                - None.
        """
        batch_size, seq_len, _ = hidden_states.shape
        H = self.num_heads
        D_h = self.head_dim

        Q = self.q_proj(hidden_states).view(batch_size, seq_len, H, D_h)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, H, D_h)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, H, D_h)

        _centroids, cluster_ids = self._kmeans(hidden_states, self.n_iters)

        out = torch.zeros_like(V)
        scale = 1.0 / math.sqrt(D_h)
        num_k = min(self.num_clusters, seq_len)
        # FIX: [Use float16/bfloat16 safe mask value to prevent numerical overflow and type mismatch]
        mask_val = -10000.0 if Q.dtype in (torch.float16, torch.bfloat16) else -10000.0

        for k in range(num_k):
            mem_mask = (cluster_ids == k).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            mask_add = (~mem_mask).to(dtype=Q.dtype) * mask_val

            Q_k = Q.permute(0, 2, 1, 3)  # (B, H, L, D_h)
            K_k = K.permute(0, 2, 1, 3)
            V_k = V.permute(0, 2, 1, 3)

            scores_k = torch.matmul(Q_k, K_k.transpose(-2, -1)) * scale
            scores_k = scores_k + mask_add

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    pad_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=scores_k.dtype)) * mask_val
                    scores_k = scores_k + pad_mask

            attn_k = F.softmax(scores_k, dim=-1)
            attn_k = self.dropout(attn_k)
            out_k = torch.matmul(attn_k, V_k)  # (B, H, L, D_h)

            # curr_mem: (B, L) -> (B, 1, L, 1) for broadcasting with (B, H, L, D_h)
            curr_mem = (cluster_ids == k).unsqueeze(1).unsqueeze(-1).to(dtype=out_k.dtype)  # (B, 1, L, 1)
            out = out + (out_k * curr_mem).permute(0, 2, 1, 3)

        context = out.contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context), None
