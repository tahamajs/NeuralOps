# framework/attentions/specialized/hamming_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hamming")
class HammingAttention(BaseAttention):
    """Hamming Attention Distillation (HAD)."""

    def __init__(self, config, rank: int = None, distill_temperature=1.0, sparsity_ratio=0.5, **kwargs):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.distill_temperature = distill_temperature
        self.sparsity_ratio = sparsity_ratio

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)  # fixed

        self.q_scale = nn.Parameter(torch.ones(self.num_heads, 1))
        self.k_scale = nn.Parameter(torch.ones(self.num_heads, 1))

    @staticmethod
    def _binarize(tensor, scale):
        t = tensor * scale.view(1, -1, 1, 1)
        return torch.tanh(t * 5.0)  # approximates sign

    @staticmethod
    def _hamming_dist(q_b, k_b):
        dot = torch.matmul(q_b, k_b.transpose(-2, -1))
        return (q_b.shape[-1] - dot) / 2.0

    def forward(self, hidden_states, attention_mask=None, teacher_attention=None, **kwargs):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        q_b = self._binarize(q, self.q_scale)
        k_b = self._binarize(k, self.k_scale)

        scores = -self._hamming_dist(q_b, k_b) / (Dh ** 0.5)

        # Mask handling (convert 2D mask to additive)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        if self.sparsity_ratio > 0:
            keep_k = max(1, int(L * self.sparsity_ratio))
            top_vals, top_idx = torch.topk(scores, k=keep_k, dim=-1)
            mask = torch.zeros_like(scores).scatter(-1, top_idx, 1.0)
            scores = scores * mask + (1 - mask) * -1e9

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        out = self.out_proj(context)

        # Always return a tuple (output, attention_weights)
        return out, None