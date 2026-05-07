import torch
import torch.nn as nn
import math
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("nsa")
class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention (NSA): Combines local sliding window with
    top‑k global sparse attention per head.
    """
    def __init__(self, config, window_size=128, top_k=32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.window_size = window_size
        self.top_k = top_k
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)

        scale = d ** -0.5

        # Local window mask
        idx = torch.arange(L, device=Q.device)
        local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)   # (1,1,L,L)

        # Global sparse: compute attention scores globally but keep only top_k per query
        # Compute full scores (quadratic) – for true NSA this would be approximated; we'll compute full and then zero out non‑top‑k.
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, H, L, L)
        # Combine local + global: we retain local window fully, plus top_k outside window
        # For each query position, get top_k outside its window
        topk_mask = torch.zeros_like(scores)
        for i in range(L):
            # get scores for query position i
            s_i = scores[:, :, i, :]   # (B, H, L)
            # mask out local window tokens
            local_indices = local_mask[0,0,i]  # (L) bool
            s_i = s_i.masked_fill(local_indices.unsqueeze(0).unsqueeze(0), float('-inf'))
            # select top_k
            _, topk_ind = torch.topk(s_i, min(self.top_k, L), dim=-1)  # (B, H, K)
            # set mask
            topk_mask[:, :, i, :] = topk_mask[:, :, i, :].scatter(2, topk_ind.unsqueeze(2), 1.0)
        # ensure local tokens are also included
        topk_mask = topk_mask + local_mask.float()
        scores = scores.masked_fill(topk_mask == 0, float('-inf'))

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(context), None