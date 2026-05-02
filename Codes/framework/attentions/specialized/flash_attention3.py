# framework/attentions/flash_attention3.py
import torch
import torch.nn as nn
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("flash3")
class FlashAttention3(BaseAttention):
    """
    FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
    For best performance use torch>=2.5, CUDA>=12.3, and H100 GPU.
    """
    def __init__(self, config, use_fp8=False,  # other args ...
        ):
        super().__init__(config)
        # Initialize projection layers
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.use_fp8 = use_fp8
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Use FlashAttention kernel
        if self.use_fp8 and hasattr(torch.nn.functional, 'scaled_dot_product_attention_fp8'):
            # Call FP8-optimized kernel
            out = torch.nn.functional.scaled_dot_product_attention_fp8(q, k, v, is_causal=False)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out), None