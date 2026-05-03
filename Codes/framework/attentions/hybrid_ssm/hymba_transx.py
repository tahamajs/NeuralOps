import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY
from ..attentions.unified_rope import UnifiedRoPE
from ..ssm.simple_ssm import SimpleSSM    # assumed available

@ATTENTION_REGISTRY.register("hymba_transx")
class HymbaTransXBlock(BaseAttention):
    """Unified RoPE, differential attention, and SSM fusion."""
    def __init__(self, config, rank=64, num_heads=None):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = num_heads or config.num_attention_heads
        self.head_dim = self.dim // self.num_heads

        self.rope = UnifiedRoPE(self.dim)
        self.W_q1 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k1 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_q2 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k2 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v  = nn.Linear(self.dim, self.dim, bias=False)
        self.mamba = SimpleSSM(self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        x_rope = self.rope(x)

        q1 = self.W_q1(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        k1 = self.W_k1(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        q2 = self.W_q2(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        k2 = self.W_k2(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        v  = self.W_v(x_rope).view(B, L, self.num_heads, self.head_dim).transpose(1,2)

        scale = self.head_dim ** -0.5
        attn1 = F.softmax(torch.matmul(q1, k1.transpose(-2,-1)) * scale, dim=-1)
        attn2 = F.softmax(torch.matmul(q2, k2.transpose(-2,-1)) * scale, dim=-1)
        diff_attn = F.relu(attn1 - 0.5 * attn2)
        A_out = torch.matmul(diff_attn, v).transpose(1,2).contiguous().view(B, L, D)

        M_out = self.mamba(x_rope)
        alpha = torch.sigmoid(self.alpha)
        out = self.out_proj(x + alpha * A_out + (1 - alpha) * M_out)
        return (out, None)