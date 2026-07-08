# framework/attentions/specialized/flash_attention3.py
import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

# Try to import FlashAttention kernels
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


@ATTENTION_REGISTRY.register("flash3")
class FlashAttention3(BaseAttention):
    """
    FlashAttention‑3: Fast and Memory‑Efficient Exact Attention.
    Uses flash_attn.flash_attn_func if installed; falls back to PyTorch SDPA.
    Paper: https://arxiv.org/abs/2205.14135
    """

    def __init__(self, config, rank: int = None, use_fp8: bool = False):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.use_fp8 = use_fp8

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, D_head = self.num_heads, self.head_dim

        q = (
            self.q_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)
        )  # (B, H, L, D_head)
        k = self.k_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, D_head).transpose(1, 2)

        if FLASH_ATTN_AVAILABLE:
            # FlashAttention expects (B, L, H, D_head) layout
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()

            if self.use_fp8 and q_fa.is_cuda:
                # Convert to FP8 (requires H100 and proper kernel)
                q_fa = q_fa.to(torch.float8_e4m3fn)
                k_fa = k_fa.to(torch.float8_e4m3fn)
                v_fa = v_fa.to(torch.float8_e4m3fn)

            out = flash_attn_func(q_fa, k_fa, v_fa, causal=False)
        else:
            # Fallback to PyTorch SDPA
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=(
                    (1 - attention_mask).to(q.dtype) * -1e9
                    if attention_mask is not None
                    else None
                ),
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), None
