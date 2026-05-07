import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(nn.Module):
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
        self.dropout = nn.Dropout(config.dropout)

    def _kmeans_assign(self, Q, centroids):
        # Q: (B, H, L, d), centroids: (B, H, K, d)
        dist = torch.cdist(Q, centroids)  # (B, H, L, K)
        return torch.argmin(dist, dim=-1)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        num_clusters = min(self.num_clusters, L)
        centroids = Q[:, :, :num_clusters, :]  # (B, H, K, d)

        for _ in range(self.n_iters):
            cluster_ids = self._kmeans_assign(Q, centroids)
            new_centroids = torch.zeros_like(centroids)
            for k in range(num_clusters):
                mask = (cluster_ids == k).unsqueeze(-1)
                sum_q = (Q * mask).sum(dim=2)
                count = mask.sum(dim=2).clamp(min=1)
                new_centroids[:, :, k, :] = sum_q / count
            centroids = new_centroids

        cluster_ids = self._kmeans_assign(Q, centroids)  # (B, H, L)
        out = torch.zeros_like(V)
        scale = d ** -0.5

        for k in range(num_clusters):
            mask = (cluster_ids == k).unsqueeze(-1)
            if mask.sum() == 0:
                continue
            Qk = Q * mask
            Kk = K * mask
            Vk = V * mask
            scores = torch.matmul(Qk, Kk.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out += torch.matmul(attn, Vk)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None