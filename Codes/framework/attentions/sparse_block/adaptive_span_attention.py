# framework/attentions/adaptive_span_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("adaptive_span")
class AdaptiveSpanAttention(nn.Module):
    """
    Adaptive Attention Span for Transformers (Facebook Research).
    Each head learns its own attention span via mask.
    """
    def __init__(self, config, max_span=1024):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.max_span = max_span
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Span parameter per head (trainable)
        self.span = nn.Parameter(torch.full((self.num_heads,), max_span, dtype=torch.float))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Compute attention with per‑head span mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * scale
        # Build causal mask limited to span length
        mask_span = torch.ones(seq_len, seq_len).tril(diagonal=self.max_span-1).bool()  # triangular
        # ...
        return out, None