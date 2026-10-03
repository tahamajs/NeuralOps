"""
T5 Architecture Compatibility Module.
Provides robust support for T5 relative position bias, encoder-decoder cross-attention,
and numerically stable FP16 forward execution for custom attention modules.
"""

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def is_t5_model(config: Any) -> bool:
    """Detect if a configuration belongs to the T5 family."""
    if config is None:
        return False
    model_type = getattr(config, 'model_type', '').lower()
    return (
        model_type == 't5'
        or hasattr(config, 'd_kv')
        or getattr(config, 'is_encoder_decoder', False)
    )


def get_t5_dims(config: Any) -> tuple[int, int, int]:
    """Extract (hidden_size, num_heads, head_dim) cleanly from T5Config or standard config."""
    hidden_size = getattr(config, 'd_model', getattr(config, 'hidden_size', 512))
    num_heads = getattr(config, 'num_heads', getattr(config, 'num_attention_heads', 8))
    head_dim = getattr(config, 'd_kv', hidden_size // num_heads if num_heads > 0 else 64)
    return hidden_size, num_heads, head_dim


def relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Translate relative positions into bucket IDs matching T5Attention._relative_position_bucket."""
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact + 1e-10)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


def compute_t5_bias(
    query_length: int,
    key_length: int,
    relative_attention_bias: nn.Embedding | None,
    is_decoder: bool = False,
    num_buckets: int = 32,
    max_distance: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Compute binned relative position bias table [1, num_heads, query_length, key_length]."""
    if relative_attention_bias is None:
        return None
    if device is None:
        device = relative_attention_bias.weight.device

    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    relative_position = memory_position - context_position

    rp_bucket = relative_position_bucket(
        relative_position,
        bidirectional=(not is_decoder),
        num_buckets=num_buckets,
        max_distance=max_distance,
    )
    values = relative_attention_bias(rp_bucket)  # [query_len, key_len, num_heads]
    values = values.permute([2, 0, 1]).unsqueeze(0)  # [1, num_heads, query_len, key_len]
    return values


class T5AttentionMixin:
    """
    Mixin providing full T5 encoder-decoder cross-attention and relative position bias
    handling for any attention class.
    """

    def init_t5_support(self, config: Any) -> None:
        """Initialize T5 attributes on module."""
        self.is_t5 = is_t5_model(config)
        self.has_relative_attention_bias = False
        self.relative_attention_bias = None
        self.relative_attention_num_buckets = getattr(config, 'relative_attention_num_buckets', 32)
        self.relative_attention_max_distance = getattr(config, 'relative_attention_max_distance', 128)
        self.is_decoder = getattr(config, 'is_decoder', False)

    def copy_t5_metadata(self, old_attn: Any) -> None:
        """Copy relative position bias embeddings and metadata from native T5 attention."""
        if old_attn is None:
            return
        self.has_relative_attention_bias = getattr(old_attn, 'has_relative_attention_bias', False)
        self.relative_attention_bias = getattr(old_attn, 'relative_attention_bias', None)
        self.relative_attention_num_buckets = getattr(old_attn, 'relative_attention_num_buckets', 32)
        self.relative_attention_max_distance = getattr(old_attn, 'relative_attention_max_distance', 128)
        self.is_decoder = getattr(old_attn, 'is_decoder', False)

    def compute_bias(self, query_length: int, key_length: int, device: torch.device | None = None) -> torch.Tensor | None:
        return compute_t5_bias(
            query_length=query_length,
            key_length=key_length,
            relative_attention_bias=self.relative_attention_bias,
            is_decoder=self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
            device=device,
        )

    def forward_t5_base(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Numerically stable T5 forward execution supporting:
        - Encoder self-attention
        - Decoder self-attention
        - Decoder cross-attention
        - Relative position bias computation and propagation
        """
        B, N_q, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states
        N_k = kv_input.shape[1]
        H = getattr(self, 'num_heads', 8)
        d = getattr(self, 'head_dim', 64)
        dtype = hidden_states.dtype
        device = hidden_states.device

        # 1. Projections
        q_mod = getattr(self, 'q_proj', getattr(self, 'q', getattr(self, 'query', None)))
        k_mod = getattr(self, 'k_proj', getattr(self, 'k', getattr(self, 'key', None)))
        v_mod = getattr(self, 'v_proj', getattr(self, 'v', getattr(self, 'value', None)))
        o_mod = getattr(self, 'out_proj', getattr(self, 'o', getattr(self, 'dense', None)))

        Q = q_mod(hidden_states).view(B, N_q, H, d).transpose(1, 2)
        K = k_mod(kv_input).view(B, N_k, H, d).transpose(1, 2)
        V = v_mod(kv_input).view(B, N_k, H, d).transpose(1, 2)

        # 2. Position bias handling (T5 standard: without sqrt(d) scaling)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        if position_bias is None:
            if getattr(self, 'has_relative_attention_bias', False) and getattr(self, 'relative_attention_bias', None) is not None:
                position_bias = self.compute_bias(N_q, N_k, device=device)
            else:
                position_bias = torch.zeros((1, H, N_q, N_k), device=device, dtype=scores.dtype)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :N_k] if attention_mask.dim() == 4 else None
                if causal_mask is not None:
                    position_bias = position_bias + causal_mask.to(scores.dtype)

        scores = scores + position_bias.to(scores.dtype)

        # 3. 2D mask handling if provided
        if attention_mask is not None and attention_mask.dim() == 2:
            min_val = torch.finfo(scores.dtype).min if scores.dtype.is_floating_point else -1e4
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2) == 0
            scores = scores.masked_fill(extended_mask, min_val)

        # 4. FP16 safe softmax in FP32
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)

        dropout = getattr(self, 'dropout', None)
        if dropout is not None and callable(dropout):
            probs = dropout(probs)

        # 5. Output projection
        context = torch.matmul(probs, V).transpose(1, 2).contiguous().view(B, N_q, -1)
        output = o_mod(context) if o_mod is not None else context
        return output, position_bias
