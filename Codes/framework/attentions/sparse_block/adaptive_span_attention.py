import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("adaptive_span")
class AdaptiveSpanAttention(nn.Module):
    """
    Adaptive Attention Span for Transformers.
    Each head learns a scalar span that soft‑masks tokens far in the future.
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
        # per‑head span parameter (softplus to ensure positivity)
        self.span_raw = nn.Parameter(torch.zeros(self.num_heads))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Learnable span: z = log(span - 1) -> span = softplus(z) + 1 to have minimum span = 1
        head_span = F.softplus(self.span_raw) + 1    # (1, H, 1, 1)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * scale

        # Causal mask with per‑head soft masking
        positions = torch.arange(seq_len, device=scores.device).float()
        relative_dist = (positions.unsqueeze(0) - positions.unsqueeze(1))
        # We only mask future tokens (j > i) by distance; we keep past intact.
        # For j > i, distance is positive; for j <= i, we set distance to 0 to keep.
        distance = torch.where(relative_dist > 0, relative_dist, torch.zeros_like(relative_dist))
        # Compute soft mask: 1 - sigmoid( (distance - span) / temp )
        # Use a small temperature to make it sharper.
        temp = 1.0
        mask_val = torch.sigmoid((distance - head_span.unsqueeze(-1)) / temp)  # (H, L, L)
        # For j <= i, we want full attention, so we force mask_val to 0 (no masking)
        mask_val = torch.where(relative_dist > 0, mask_val, torch.zeros_like(mask_val))
        # Add a huge negative before softmax: subtract 1e9 * mask_val
        scores = scores - 1e9 * mask_val.unsqueeze(0).unsqueeze(0)  # broadcast to (1, H, L, L)

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, V.permute(0,2,1,3))  # (B, H, L, d)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(context)
        return out, attn_probs