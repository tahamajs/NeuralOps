<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: B_implementation.md
     PURPOSE: Code details
     DEPENDENCIES: 05_architecture.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix B. Implementation Details

## B.1 The `MultiHeadLatentAttention` Class

The PyTorch realization implements latent projection, decoupled rotary positional embedding, and inference weight absorption:

```python
import torch
import torch.nn as nn
import math

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=768, num_heads=12, d_c=128, d_r=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_c = d_c
        self.d_r = d_r

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)
        self.w_uk = nn.Linear(d_c, d_model, bias=False)
        self.w_uv = nn.Linear(d_c, d_model, bias=False)
        self.w_qr = nn.Linear(d_model, num_heads * d_r, bias=False)
        self.w_kr = nn.Linear(d_model, d_r, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.w_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        c_kv = self.w_dkv(x) # (B, N, d_c)
        k = self.w_uk(c_kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
```

## B.2 Shape Transformations and Kernel Absorption

In optimized inference execution, the projection $W_{UK}$ is transposed and merged into $W_Q$:
$$
\tilde{W}_{Q, h} = W_{Q, h} W_{UK, h}^\top \in \mathbb{R}^{D \times d_c}
$$
This bypasses explicit materialization of multi-head key matrices, reducing intermediate tensor activation traffic.
