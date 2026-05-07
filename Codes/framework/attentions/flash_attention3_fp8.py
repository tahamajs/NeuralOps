import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

try:
    from flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

@ATTENTION_REGISTRY.register("flash3_fp8")
class FlashAttention3FP8(nn.Module):
    def __init__(self, d_model, n_heads, use_fp8=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_fp8 = use_fp8
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, **kwargs):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)
        if FLASH_ATTN_AVAILABLE and self.use_fp8 and q.is_cuda:
            q = q.to(torch.float8_e4m3fn)
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)
            out = flash_attn_func(q, k, v, causal=False).to(x.dtype)
        else:
            from torch.nn.functional import scaled_dot_product_attention
            out = scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out_proj(out.reshape(B, L, D))

@ATTENTION_REGISTRY.register("flash3_fp8")
def inject_flash3_fp8(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    use_fp8 = getattr(comp_config, 'use_fp8', True)
    new_attn = FlashAttention3FP8(d_model=model_config.hidden_size, n_heads=model_config.num_attention_heads, use_fp8=use_fp8)
    # optionally copy weights from old_attn
    return new_attn