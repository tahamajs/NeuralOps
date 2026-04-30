import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
    from flash_attn_interface import flash_attn_func as flash_attn_fp8
except ImportError:
    raise ImportError("Install flash_attn and flash_attn_interface for FlashAttention 3 support")

class FlashAttention3Block(nn.Module):
    """Replace standard attention with FlashAttention 3 in FP8."""
    def __init__(self, d_model, n_heads, causal=True, use_fp8=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.causal = causal
        self.use_fp8 = use_fp8

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        if self.use_fp8 and q.is_cuda:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
            out = flash_attn_fp8(q, k, v, causal=self.causal)
            out = out.to(x.dtype)
        else:
            out = flash_attn_func(q, k, v, causal=self.causal)
        out = out.reshape(B, L, D)
        return self.out_proj(out)