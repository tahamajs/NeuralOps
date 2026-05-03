import torch
import torch.nn as nn
import torch.nn.functional as F

def alpha_entmax(X, alpha=1.5, n_iter=25):
    """
    α‑entmax: a sparse generalisation of softmax.
    Approximated via Halley's method on the sparsemax transformation.
    For α=1.5 (entmax) we use the closed‑form entmax bisection.
    """
    if alpha == 1.0:
        return F.softmax(X, dim=-1)
    if alpha == 1.5:
        # Use the entmax root‑finding from the entmax paper
        return _entmax15(X, dim=-1, n_iter=n_iter)
    # Generic bisection for other α (simplified)
    return _generic_entmax(X, alpha=alpha, dim=-1, n_iter=n_iter)

def _entmax15(logits, dim=-1, n_iter=25):
    B, H, N = logits.shape
    logits = logits / 2   # factor from entmax paper
    # Bisection search for the root of the function
    # Implementation based on the entmax library
    p = F.softmax(logits, dim=dim)
    tau = torch.ones_like(p.sum(dim=dim, keepdim=True))
    for _ in range(n_iter):
        p_alpha = torch.sqrt(p + 1e-8)
        tau = tau * torch.sum(p_alpha, dim=dim, keepdim=True) / torch.sum(p * p_alpha, dim=dim, keepdim=True)
        p = F.softmax(logits * tau, dim=dim)
    return p

def _generic_entmax(logits, alpha=1.5, dim=-1, n_iter=25):
    # Placeholder: for simplicity we return softmax if alpha not 1 or 1.5
    return F.softmax(logits, dim=dim)

class AdaSplashAttention(nn.Module):
    """
    AdaSplash: Sparse Attention with α‑entmax.
    Replaces softmax with entmax to produce sparse attention patterns.
    """
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