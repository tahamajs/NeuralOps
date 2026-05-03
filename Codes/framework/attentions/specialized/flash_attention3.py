import torch
import torch.nn as nn
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("flash3")
class FlashAttention3(BaseAttention):
    """
    FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.
    Uses flash_attn_func (or torch's SDPA if not available).
    """
    def __init__(self, config, use_fp8=False):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.use_fp8 = use_fp8

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        q = self.query(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(B, L, self.num_heads, self.head_dim)

        # Try to use flash-attention if available
        try:
            from flash_attn import flash_attn_func
            out = flash_attn_func(q, k, v, causal=False)
        except ImportError:
            # Fallback to PyTorch's SDPA
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                is_causal=False
            ).transpose(1,2)

        out = out.contiguous().view(B, L, D)
        return self.out_proj(out), None