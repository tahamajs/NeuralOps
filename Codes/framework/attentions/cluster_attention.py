import math

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("cluster")
class ClusterAttention(nn.Module):
    """
    ClusterFormer: Token clustering via k‑means to form cluster centroids,
    then intra‑cluster attention within each cluster.
    """

    def __init__(self, config, num_clusters=64, n_iters=3):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_clusters = num_clusters
        self.n_iters = n_iters
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _kmeans(self, x, iters=3):
        """Batch‑wise k‑means for clustering."""
        B, L, D = x.shape
        # Initialise centroids with random sampling from x
        idx = torch.randperm(L)[: self.num_clusters]
        centroids = x[:, idx, :]  # (B, K, D)
        for _ in range(iters):
            # Compute distances: (B, L, K)
            dist = torch.cdist(x, centroids)
            cluster_ids = torch.argmin(dist, dim=-1)  # (B, L)
            # Update centroids: mean of points in each cluster
            centroids = torch.zeros_like(centroids)
            for k in range(self.num_clusters):
                mask_k = (cluster_ids == k).unsqueeze(-1).float()
                sum_k = (x * mask_k).sum(dim=1, keepdim=True)
                count_k = mask_k.sum(dim=1, keepdim=True).clamp(min=1)
                centroids[:, k, :] = (sum_k / count_k).squeeze(1)
        return centroids, cluster_ids

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        H = self.num_heads
        D_h = self.head_dim

        Q = self.q_proj(hidden_states).view(batch_size, seq_len, H, D_h)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, H, D_h)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, H, D_h)

        # For clustering we use the original hidden states (or Q) to form groups.
        # We cluster per head using Q mean across heads for simplicity.
        # Here we cluster tokens globally (same clusters for all heads) using mean of Q.
        Q.mean(
            dim=2
        )  # (B, L, D_h) – use channel‑wise mean across heads? Better to use hidden_states.
        # Actually, use the hidden_states directly for clustering.
        centroids, cluster_ids = self._kmeans(
            hidden_states, self.n_iters
        )  # cluster_ids: (B, L)

        # For each cluster, compute attention within the cluster.
        # We'll do per head and per batch, but for simplicity we loop over clusters.
        out = torch.zeros_like(V)
        scale = 1.0 / math.sqrt(D_h)
        for k in range(self.num_clusters):
            # get mask for tokens in this cluster
            mask = (cluster_ids == k).unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            # Select Q, K, V for this cluster
            Q_k = Q * mask.unsqueeze(2).float()
            K_k = K * mask.unsqueeze(2).float()
            V_k = V * mask.unsqueeze(2).float()

            # Attention within cluster (causal? we just do full attention within cluster, no causal needed for encoding)
            # However, for language models we need causal; we assume not for simplicity (or we can add later).
            # We'll do full attention inside the cluster.
            scores_k = torch.einsum("blhd,bmhd->bhlm", Q_k, K_k) * scale
            # Mask out tokens not in this cluster by setting scores to -inf where token not in cluster.
            # But Q_k already zero for non-members, scores would be zero for those, still they could attend.
            # Better: create additive mask.
            # We'll manually set scores to -inf for non‑member queries and keys.
            # For simplicity, after softmax, the contributions from zero queries/keys will be zero anyway because they are zero.
            # We'll just proceed with masking by -inf.
            # Build a mask using cluster_ids
            mem_mask = (cluster_ids == k).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            # Expand to (B, H, L, L) by broadcasting
            mask_add = (~mem_mask).float() * -1e9
            scores_k = scores_k + mask_add
            attn_k = F.softmax(scores_k, dim=-1)
            attn_k = self.dropout(attn_k)
            out_k = torch.matmul(attn_k, V_k)  # (B, H, L, D_h)
            out += out_k

        context = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context), None
