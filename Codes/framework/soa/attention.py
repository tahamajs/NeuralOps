import torch
import torch.nn as nn
import math

class SOAFusedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k_rank):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank

        self.q_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_latent, attn_mask=None):
        # x_latent: (B, S, k_rank)
        B, S, _ = x_latent.shape
        q = self.q_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        return self.o_proj(out)