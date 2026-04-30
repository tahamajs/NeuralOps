# framework/attentions/vig_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViGAttention(nn.Module):
    """
    Direction‑wise gated linear attention for images (2D grid).
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_g = nn.Linear(dim, 2*num_heads)   # for row & col directions
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, D = x.shape  # N = H*W
        q = self.W_q(x).view(B, N, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, N, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, N, self.num_heads, self.head_dim)
        gates = torch.sigmoid(self.W_g(x)).view(B, N, 2, self.num_heads, 1)
        # Row‑wise linear attention
        q_row = q.reshape(B, H, W, self.num_heads, self.head_dim).permute(0,2,1,3,4)  # (B,W,H,Hdim)
        k_row = k.reshape(...)
        v_row = v.reshape(...)
        kv_row = torch.einsum("bwihd,bwihe->bwihde", k_row, v_row)
        out_row = torch.einsum("bwihd,bwihe->bwihe", q_row, kv_row)
        # Col‑wise similarly
        # Combine with learned gates
        out = gates[:,:,:,0] * out_row + gates[:,:,:,1] * out_col
        out = out.reshape(B, N, D)
        return self.out_proj(out)