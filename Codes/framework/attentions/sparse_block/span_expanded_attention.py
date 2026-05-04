# framework/attentions/specialized/span_expanded_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("span_expanded")
class SpanExpandedAttention(BaseAttention):
    """
    Span‑Expanded Attention with HyLoRA: fixed expansion cache, efficient LoRA adaptation.
    """
    def __init__(self, config, rank: int = None, expansion_span: int = 1024,
                 use_lora: bool = True, lora_rank: int = 8):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.expansion_span = expansion_span
        self.use_lora = use_lora

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        if use_lora:
            self.lora_A = nn.Linear(self.dim, lora_rank, bias=False)
            self.lora_B = nn.Linear(lora_rank, self.dim, bias=False)

    def forward(self, hidden_states, kv_cache=None, **kwargs):
        B, L, D = hidden_states.shape
        q = self.W_q(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, L, d)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = k_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)   # (B, H, C, d)
            v = v_cache.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            k = self.W_k(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.W_v(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                   # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        if self.use_lora:
            lora_out = self.lora_B(self.lora_A(hidden_states))        # (B, L, D)
            out = out + lora_out

        out = self.out_proj(out)
        return out, None   # return tuple as required by attention interface