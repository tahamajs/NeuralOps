#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Cross-Attention Layer.

Multi-head cross-attention mechanism enabling query tokens from one sequence to attend
to key and value tokens from another sequence (e.g. encoder-decoder architecture).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


class CrossAttention(BaseAttention):
    """Multi-head cross-attention module with LoRA projections.

    Attributes:
        supports_causal: Whether causal masking is supported when operating as self-attention.
        supports_bidirectional: Whether bidirectional attention is supported.
        supports_gqa: Whether grouped-query head broadcasting is supported.
    """

    FAMILY: str = "modern"
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        rank: int | None = None,
        embed_dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        lora_rank: int = 16,
        is_decoder: bool | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes CrossAttention."""
        try:
            super().__init__(config=config, rank=rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")
        resolved_dim = (
            embed_dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "embed_dim", None)
            or _cfg_get(config, "dim", 768)
        )
        if resolved_dim is None and src is not None:
            resolved_dim = getattr(src, "hidden_size", getattr(src, "embed_dim", 768))
        self.embed_dim: int = int(resolved_dim or 768)
        self.hidden_size: int = self.embed_dim
        self.dim: int = self.embed_dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", 8)
        )
        if resolved_heads is None and src is not None:
            resolved_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", 8))
        self.num_heads: int = max(1, int(resolved_heads or 8))

        resolved_kv_heads = (
            num_kv_heads
            or kwargs.get("num_key_value_heads")
            or _cfg_get(config, "num_key_value_heads", None)
            or _cfg_get(config, "num_kv_heads", self.num_heads)
        )
        self.num_kv_heads: int = max(1, min(int(resolved_kv_heads or self.num_heads), self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)

        resolved_head_dim = head_dim or kwargs.get("head_dim") or (self.embed_dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.is_decoder: bool = bool(is_decoder if is_decoder is not None else _cfg_get(config, "is_decoder", False))

        self.lora_rank: int = int(getattr(config, "lora_rank", lora_rank))
        self.q_proj = LoRALinear(
            self.embed_dim,
            self.all_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.embed_dim,
            self.kv_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.embed_dim,
            self.kv_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.embed_dim,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )

        dropout_p = float(dropout or _cfg_get(config, "dropout", 0.0))
        self.dropout_p: float = dropout_p
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.scale: float = 1.0 / math.sqrt(max(self.head_dim, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs kwargs like key_value_states
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for cross-attention.

        Args:
            hidden_states: Query sequence tensor of shape (batch, seq_len_q, dim).
            context: Key/value sequence tensor of shape (batch, seq_len_kv, dim).
                     If None, checks encoder_hidden_states, or falls back to self-attention.
            attention_mask: Optional mask for query sequence.
            encoder_hidden_states: Alternate name for context.
            encoder_attention_mask: Mask for context sequence.
            output_attentions: Whether to compute and return attention weights.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output, attention_weights).
        """
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        kv = context
        if kv is None:
            kv = encoder_hidden_states
        if kv is None:
            kv = kwargs.get("key_value_states")
        is_cross = kv is not None and kv is not hidden_states
        if not is_cross:
            kv = hidden_states

        b, n_q, _ = hidden_states.shape
        n_kv = kv.shape[1]
        h, h_kv, d = self.num_heads, self.num_kv_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(b, n_q, h, d).transpose(1, 2).contiguous()
        k = self.k_proj(kv).view(b, n_kv, h_kv, d).transpose(1, 2).contiguous()
        v = self.v_proj(kv).view(b, n_kv, h_kv, d).transpose(1, 2).contiguous()

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross and encoder_attention_mask is not None else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("mask")

        is_causal = (not is_cross) and self.is_decoder and n_q > 1

        use_sdpa = not output_attentions
        context_out: torch.Tensor | None = None
        attn_weights: torch.Tensor | None = None

        if use_sdpa:
            try:
                attn_bias = None
                if eff_mask is not None:
                    if eff_mask.dim() == 4:
                        attn_bias = eff_mask.to(q.dtype)
                    elif eff_mask.dim() == 2:
                        attn_bias = ((eff_mask[:, None, None, :].float() - 1.0) * 1e4).to(q.dtype)

                dropout_val = self.dropout_p if self.training else 0.0
                context_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    is_causal=is_causal and attn_bias is None,
                    dropout_p=dropout_val,
                )
            except Exception as e_sdpa:
                logger.debug("CrossAttention SDPA failed: %s", e_sdpa)
                use_sdpa = False

        if not use_sdpa or context_out is None:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            if is_causal:
                q_idx = torch.arange(n_q, device=device).unsqueeze(1) + (n_kv - n_q)
                k_idx = torch.arange(n_kv, device=device).unsqueeze(0)
                scores = scores.masked_fill((k_idx > q_idx).unsqueeze(0).unsqueeze(0), -1e4)

            if eff_mask is not None:
                if eff_mask.dim() == 4:
                    scores = scores + eff_mask.to(scores.dtype)
                else:
                    valid_m = get_token_valid_mask(eff_mask, b, n_kv, device)
                    if valid_m is not None:
                        scores = scores.masked_fill(~valid_m[:, None, None, :], -1e4)

            attn_weights = F.softmax(scores, dim=-1).to(dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            context_out = torch.matmul(attn_weights, v)

        out = context_out.transpose(1, 2).contiguous().view(b, n_q, self.all_head_size)
        out = self.out_proj(out)
        return out, attn_weights


cross_attention = CrossAttention

__all__ = [
    "CrossAttention",
    "cross_attention",
]
