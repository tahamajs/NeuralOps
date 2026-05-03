import torch
import torch.nn as nn
import torch.nn.functional as F

class SpanExpandedAttention(nn.Module):
    """
    Span‑Expanded Attention with HyLoRA: fixed expansion cache, efficient LoRA adaptation.
    """
    def __init__(self, dim: int, num_heads: int = 16, expansion_span: int = 1024,
                 use_lora: bool = True, lora_rank: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.expansion_span = expansion_span
        self.use_lora = use_lora

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        if use_lora:
            self.lora_A = nn.Linear(dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: tuple = None) -> torch.Tensor:
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, L, d)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = k_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)   # (B, H, C, d)
            v = v_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                   # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        if self.use_lora:
            lora_out = self.lora_B(self.lora_A(x))                    # (B, L, D)
            out = out + lora_out

        return self.out_proj(out)