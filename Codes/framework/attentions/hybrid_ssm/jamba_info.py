import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

class InfoMambaConceptBottleneck(nn.Module):
    """Minimal‑bandwidth global interface: concept bottleneck."""
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.W_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.W_up = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x):
        return self.W_up(F.gelu(self.W_down(x)))

@ATTENTION_REGISTRY.register("jamba_info")
class JambaInfoBlock(BaseAttention):
    """
    Jamba‑Info segregated block.
    Retrieval mode: differential attention with sparse heads.
    Compression mode: concept bottleneck + SSM.
    """
    def __init__(self, config, rank=64, mode="retrieval", sparsity=0.15):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.mode = mode

        if mode == "retrieval":
            self.num_heads = config.num_attention_heads
            self.head_dim = self.dim // self.num_heads
            self.active_heads = max(1, int(self.num_heads * sparsity))
            self.W_q1 = nn.Linear(self.dim, self.dim, bias=False)
            self.W_k1 = nn.Linear(self.dim, self.dim, bias=False)
            self.W_q2 = nn.Linear(self.dim, self.dim, bias=False)
            self.W_k2 = nn.Linear(self.dim, self.dim, bias=False)
            self.W_v = nn.Linear(self.dim, self.dim, bias=False)
            self.out_proj = nn.Linear(self.dim, self.dim)
        else:
            self.bottleneck = InfoMambaConceptBottleneck(self.dim)
            from ..ssm.simple_ssm import SimpleSSM
            self.ssm = SimpleSSM(self.dim)
            self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        if self.mode == "retrieval":
            q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
            k1 = self.W_k1(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
            q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
            k2 = self.W_k2(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
            v  = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)

            scale = self.head_dim ** -0.5
            attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2,-1)) * scale, dim=-1)
            attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2,-1)) * scale, dim=-1)
            diff_attn = attn1 - 0.5 * attn2
            diff_attn[:, self.active_heads:] = 0   # sparsify lower heads
            out = torch.matmul(diff_attn, v).transpose(1,2).contiguous().view(B, L, D)
        else:
            x_bottleneck = self.bottleneck(x)
            ssm_out = self.ssm(x_bottleneck)
            out = x + ssm_out
        return self.out_proj(out), None