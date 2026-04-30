# framework/attentions/adasplash_attention.py
import torch
import torch.nn as nn

def alpha_entmax(X, alpha=1.5, n_iter=7):
    # Simplified Halley‑bisection implementation (the real AdaSplash uses Triton)
    def bisection_step(p, alpha, tau):
        # Placeholder for the Halley‑bisection update
        return torch.clamp(p, 0, 1)
    # Use softmax as initial guess
    p = torch.softmax(X, dim=-1)
    tau = torch.zeros_like(p.sum(dim=-1, keepdim=True))
    for _ in range(n_iter):
        p = bisection_step(p, alpha, tau)
    return p

class AdaSplashAttention(nn.Module):
    def __init__(self, dim, num_heads, alpha=1.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.alpha = alpha
        self.qkv = nn.Linear(dim, 3*dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / (self.head_dim**0.5)
        attn = alpha_entmax(scores, self.alpha)
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(out)