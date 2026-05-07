import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from .simple_ssm import SimpleSSM    # assumed available
import torch
import torch.nn as nn

class UnifiedRoPE(nn.Module):
    """
    Unified Rotary Position Embedding (RoPE) that can be applied to any hidden states.
    Rotates half of the feature dimensions based on position indices.
    """
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency bands: shape (dim//2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin tables for all positions up to max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """Precompute cos and sin for positions 0..seq_len-1."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)          # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(2))   # (1, seq_len, 1, dim)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(2))
    
    def forward(self, x, offset=0):
        """
        Apply RoPE to input tensor.
        
        Args:
            x: (batch_size, seq_len, hidden_dim)
            offset: starting position offset (for incremental decoding)
        Returns:
            Rotated tensor of same shape.
        """
        batch, seq_len, dim = x.shape
        assert dim == self.dim, f"Expected input dim {self.dim}, got {dim}"
        
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        # Get cos and sin for the required positions
        cos = self.cos_cached[:, offset:offset+seq_len, :, :]   # (1, seq_len, 1, dim)
        sin = self.sin_cached[:, offset:offset+seq_len, :, :]
        
        # Reshape input to separate even and odd channels
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Stack to apply rotation: (x_even, x_odd) -> (x_even * cos - x_odd * sin, x_odd * cos + x_even * sin)
        rotated_even = x_even * cos.squeeze(2)[..., 0::2] - x_odd * sin.squeeze(2)[..., 0::2]
        rotated_odd = x_odd * cos.squeeze(2)[..., 0::2] + x_even * sin.squeeze(2)[..., 0::2]
        # Interleave back
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        return rotated
    
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