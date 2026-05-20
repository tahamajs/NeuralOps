# framework/attentions/specialized/routing_attention.py
"""
Routing Attention: K‑means clustering of queries for sparse attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(nn.Module):
    """
    Routing Attention that clusters queries via k‑means and performs
    intra‑cluster attention. Clustering is separate per head.
    """

    def __init__(self, config, num_clusters: int = 32, n_iters: int = 3,
                 pretrained_attn: nn.Module = None):
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

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
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

    def _kmeans_assign(self, Q, centroids):
        # Q: (B, H, L, d), centroids: (B, H, K, d)
        dist = torch.cdist(Q, centroids)   # (B, H, L, K)
        return torch.argmin(dist, dim=-1)  # (B, H, L)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        # Pad mask for valid tokens
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(Q.device)   # (B, L)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
            # Set queries/keys/values of padding tokens to zero
            valid_4d = valid.unsqueeze(1).unsqueeze(-1).to(Q.dtype)   # (B, 1, L, 1)
            Q = Q * valid_4d
            K = K * valid_4d
            V = V * valid_4d
        else:
            valid = None

        num_clusters = min(self.num_clusters, L)
        # Initial centroids from first K queries
        centroids = Q[:, :, :num_clusters, :]   # (B, H, K, d)

        for _ in range(self.n_iters):
            cluster_ids = self._kmeans_assign(Q, centroids)
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(num_clusters):
                mask = (cluster_ids == k).unsqueeze(-1)   # (B, H, L, 1)
                sum_q = (Q * mask).sum(dim=2)
                count = mask.sum(dim=2).clamp(min=1)
                new_centroids[:, :, k, :] = sum_q / count
            centroids = new_centroids

        cluster_ids = self._kmeans_assign(Q, centroids)   # (B, H, L)
        out = torch.zeros_like(V)
        scale = d ** -0.5

        for k in range(num_clusters):
            mask = (cluster_ids == k).unsqueeze(-1)   # (B, H, L, 1)
            if mask.sum() == 0:
                continue
            # Zero out non‑members
            Qk = Q * mask
            Kk = K * mask
            Vk = V * mask
            scores = torch.matmul(Qk, Kk.transpose(-2, -1)) * scale
            # Mask padding in keys (already zero, but softmax may still see zero scores)
            if valid is not None:
                key_valid = valid.unsqueeze(1).unsqueeze(-2).to(scores.dtype)   # (B, 1, 1, L)
                scores = scores.masked_fill(key_valid == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out += torch.matmul(attn, Vk)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, None   # We don't return full attention (could be added)
        return out, None


def inject_routing(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    num_clusters = getattr(comp_config, 'num_clusters', 32)
    n_iters = getattr(comp_config, 'n_iters', 3)
    new_attn = RoutingAttention(config=model_config, num_clusters=num_clusters,
                                n_iters=n_iters, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn