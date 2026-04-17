import torch
import torch.nn as nn
import math

class SOAFusedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k_rank):
        super().__init__()
        self.hidden = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.qkv = nn.Linear(k_rank, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_latent, attn_mask):
        # x_latent: (B,S,r)
        B, S, _ = x_latent.shape
        qkv = self.qkv(x_latent)              # (B,S,3H)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1,2)  # (B,nh,S,hd)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.head_dim)  # (B,nh,S,S)
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # (B,nh,S,hd)
        ctx = ctx.transpose(1,2).contiguous().view(B, S, self.hidden)

        return self.out(ctx)
