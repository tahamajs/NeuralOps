# framework/soa/attention.py
import torch
import torch.nn as nn
import math

class SOAFusedAttention(nn.Module):
    """
    Fused attention that works on low‑rank latent representations.
    """
    def __init__(self, hidden_size: int, num_heads: int, k_rank: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank

        # Project from latent space to Q, K, V
        self.q_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_latent: torch.Tensor, attn_mask: torch.Tensor = None):
        B, S, _ = x_latent.shape

        q = self.q_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.out_proj(context)