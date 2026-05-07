import torch
import torch.nn as nn
import math
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(nn.Module):
    """
    Routing Transformer: Attention via online k‑means clustering.
    Queries are assigned to clusters, and attention is computed only within each cluster.
    """
    def __init__(self, config, num_clusters=32, n_iters=3):
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

    def _kmeans_assign(self, Q, centroids):
        # Q: (B, H, L, d), centroids: (B, H, K, d)
        dist = torch.cdist(Q, centroids)  # (B, H, L, K)
        return torch.argmin(dist, dim=-1)  # (B, H, L)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        # Project and reshape to multi‑head
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)  # (B, H, L, d)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        # Initialise centroids with first K tokens (or random)
        centroids = Q[:, :, :self.num_clusters, :]  # (B, H, K, d)

        # Online k‑means refinement
        for _ in range(self.n_iters):
            cluster_ids = self._kmeans_assign(Q, centroids)   # (B, H, L)
            # Update centroids as mean of assigned queries
            centroids = torch.zeros_like(centroids)
            for k in range(self.num_clusters):
                mask = (cluster_ids == k).unsqueeze(-1).float()  # (B, H, L, 1)
                sum_q = (Q.unsqueeze(2) * mask.unsqueeze(2)).sum(dim=3, keepdim=True)  # (B, H, 1, d)
                count = mask.sum(dim=2, keepdim=True).clamp(min=1)
                centroids[:, :, k, :] = (sum_q / count).squeeze(2)

        # Final assignment
        cluster_ids = self._kmeans_assign(Q, centroids)

        # Intra‑cluster attention
        out = torch.zeros_like(V)
        scale = d ** -0.5

        for k in range(self.num_clusters):
            mask = (cluster_ids == k).unsqueeze(-1)  # (B, H, L, 1)

            # Apply mask to Q, K, V (set non‑members to 0 – not necessary, we'll mask scores)
            Q_k = Q * mask
            K_k = K * mask
            V_k = V * mask

            # Scores within cluster (mask out tokens from other clusters)
            scores = torch.matmul(Q_k, K_k.transpose(-2, -1)) * scale  # (B, H, L, L)
            # Add large negative value where tokens do not belong to the same cluster
            mask_add = (mask == 0).float() * -1e9  # (B, H, L, 1) broadcast to (B, H, L, L)
            scores = scores + mask_add.squeeze(-1).unsqueeze(-2)  # + (B, H, 1, L) effectively

            # Optional: causal masking? Uncomment if needed for language models
            # causal_mask = torch.tril(torch.ones(L, L, device=scores.device)).bool()
            # scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_k = torch.matmul(attn, V_k)  # (B, H, L, d)
            out = out + out_k

        # Combine heads and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None