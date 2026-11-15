#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Flash-Decoding Attention Layer.

Accelerates autoregressive token decoding (N_q = 1) over long sequence contexts via
Split-KV chunking and multi-block parallel reduction, with strict past_key_values cache threading.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_utils import LoRALinear
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def flash_decoding_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Dispatches split-KV flash decoding with Triton acceleration or PyTorch reference.

    Args:
        q: Query tensor (batch, heads, 1, head_dim) or (batch, heads, head_dim).
        k: Key tensor (batch, heads, seq_len_kv, head_dim).
        v: Value tensor (batch, heads, seq_len_kv, head_dim).
        chunk_size: Token chunk size for parallel reduction.

    Returns:
        Decoded context tensor matching query layout.
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)

    # Fast hardware path via Triton when on CUDA
    if q.is_cuda and k.is_cuda and v.is_cuda:
        try:
            from neuralops.ops.triton.flash_decoding import flash_decoding_attention

            return flash_decoding_attention(q, k, v, chunk_size=chunk_size)
        except Exception as e_triton:
            logger.debug("Flash decoding Triton dispatch fallback: %s", e_triton)

    # Reference chunked online softmax
    b, h, n_q, d = q.shape
    n_kv = k.shape[-2]
    scale = 1.0 / math.sqrt(max(d, 1))

    if n_kv <= chunk_size:
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1).to(q.dtype)
        return torch.matmul(weights, v)

    num_chunks = (n_kv + chunk_size - 1) // chunk_size
    chunk_outs = []
    chunk_lses = []

    for i in range(num_chunks):
        st = i * chunk_size
        en = min(st + chunk_size, n_kv)
        k_c = k[:, :, st:en, :]
        v_c = v[:, :, st:en, :]

        s_c = torch.matmul(q.float(), k_c.float().transpose(-2, -1)) * scale
        lse_c = torch.logsumexp(s_c, dim=-1, keepdim=True)
        w_c = torch.exp(s_c - lse_c).to(q.dtype)
        out_c = torch.matmul(w_c, v_c)

        chunk_outs.append(out_c)
        chunk_lses.append(lse_c)

    all_lse = torch.cat(chunk_lses, dim=-1)
    global_lse = torch.logsumexp(all_lse, dim=-1, keepdim=True)

    final_out = torch.zeros_like(q)
    for out_c, lse_c in zip(chunk_outs, chunk_lses):
        alpha = torch.exp(lse_c - global_lse).to(q.dtype)
        final_out = final_out + alpha * out_c

    return final_out


@ATTENTION_REGISTRY.register(
    "flash_decoding",
    supports_causal=True,
    supports_bidirectional=False,
    supports_gqa=True,
)
class FlashDecodingAttention(BaseAttention):
    """Flash-Decoding Attention layer with strict past_key_values cache threading.

    Attributes:
        supports_causal: True.
        supports_bidirectional: False.
        supports_gqa: True.
    """

    FAMILY: str = "fast_kernels"
    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace config or dict
        dim: int | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        chunk_size: int = 128,
        bias: bool = True,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes FlashDecodingAttention."""
        super().__init__(config=config)

        resolved_dim = (
            dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", 768)
        )
        self.dim: int = int(resolved_dim)
        self.hidden_size: int = self.dim

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", 8)
        )
        self.num_heads: int = max(1, int(resolved_heads))

        resolved_head_dim = head_dim or (self.dim // self.num_heads)
        self.head_dim: int = int(resolved_head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.chunk_size: int = int(chunk_size)

        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.all_head_size, bias=bias, freeze_base=True)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=bias, freeze_base=True)

    @staticmethod
    def _read_cache(
        past_key_value: Any,  # reason: tuple or DynamicCache instance
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extracts cached (K, V) tensors for this layer."""
        if past_key_value is None:
            return None, None

        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) == 0:
                return None, None
            first = past_key_value[0]
            if isinstance(first, (tuple, list)):
                if layer_idx < len(past_key_value):
                    entry = past_key_value[layer_idx]
                    if len(entry) >= 2 and entry[0] is not None:
                        return entry[0], entry[1]
                return None, None
            if len(past_key_value) >= 2 and past_key_value[0] is not None:
                return past_key_value[0], past_key_value[1]
            return None, None

        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            if layer_idx < len(past_key_value.key_cache):
                k = past_key_value.key_cache[layer_idx]
                v = past_key_value.value_cache[layer_idx] if hasattr(past_key_value, "value_cache") else None
                return k, v

        return None, None

    @staticmethod
    def _write_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Any,  # reason: tuple or DynamicCache instance
        layer_idx: int,
    ) -> Any:  # reason: updated tuple or DynamicCache instance
        """Updates and threads past_key_values cache."""
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                return past_key_value
            except Exception:
                pass

        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if hasattr(past_key_value, "value_cache"):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                if hasattr(past_key_value, "value_cache"):
                    past_key_value.value_cache.append(v)
            return past_key_value

        return (k, v)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Any | None = None,  # reason: HF KV cache structure
        *args: Any,  # reason: absorbs unused positional args
        **kwargs: Any,  # reason: absorbs layer_idx, use_cache
    ) -> tuple[torch.Tensor, Any | None]:
        """Forward pass with split-KV flash decoding and cache threading."""
        b, seq_len_q, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim
        layer_idx = int(kwargs.get("layer_idx", 0))

        q = self.q_proj(hidden_states).view(b, seq_len_q, h, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, seq_len_q, h, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, seq_len_q, h, d).transpose(1, 2)

        # Thread past_key_values
        past_k, past_v = self._read_cache(past_key_value, layer_idx)
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k.to(k.dtype), k], dim=-2)
            v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        present_key_value: Any | None = None
        if past_key_value is not None or kwargs.get("use_cache", False):
            present_key_value = self._write_cache(k, v, past_key_value, layer_idx)

        # If decoding single token (seq_len_q == 1), use Split-KV flash decoding
        if seq_len_q == 1:
            context = flash_decoding_forward(q, k, v, chunk_size=self.chunk_size)
        else:
            # Prefill phase: standard causal attention
            context = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        context = context.transpose(1, 2).contiguous().view(b, seq_len_q, self.all_head_size)
        out = self.out_proj(context)
        return out, present_key_value


FlashDecoding = FlashDecodingAttention
flash_decoding = FlashDecodingAttention

__all__ = [
    "FlashDecoding",
    "FlashDecodingAttention",
    "flash_decoding",
    "flash_decoding_forward",
]
