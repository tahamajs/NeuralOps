#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Scaled Dot-Product Attention (SDPA Attention) Module.

High-level attention layer wrapping PyTorch SDPA with automatic hardware backend
selection, stateful KV caching, GQA expansion, and post-projection rotary embedding hooks.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.sdpa import SDPABackend, sdpa_forward

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "sdpa_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
@ATTENTION_REGISTRY.register(
    "sdpa",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class SDPAAttention(BaseAttention):
    """Scaled Dot-Product Attention layer with backend dispatch.

    Attributes:
        supports_causal: Whether causal autoregressive masking is supported.
        supports_bidirectional: Whether bidirectional non-causal attention is supported.
        supports_gqa: Whether grouped/multi-query broadcasting is supported.
        supports_rope: Whether rotary position embeddings are supported.
        is_custom_attention: Flag indicating custom attention implementation.
    """

    FAMILY: str = "dense"
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,  # reason: HuggingFace PretrainedConfig or dict
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool | None = None,
        preferred_backend: SDPABackend | str = SDPABackend.AUTO,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        hidden_dim: int | None = None,
        is_causal: bool | None = None,
        **kwargs: Any,  # reason: arbitrary hyperparameters
    ) -> None:
        """Initializes SDPAAttention."""
        try:
            super().__init__(config=config, rank=rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")
        resolved_dim = (
            dim
            or hidden_dim
            or kwargs.get("hidden_dim")
            or kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or kwargs.get("embed_dim")
            or _cfg_get(config, "hidden_dim", None)
            or _cfg_get(config, "hidden_size", None)
            or _cfg_get(config, "dim", None)
            or _cfg_get(config, "d_model", None)
        )
        if resolved_dim is None and src is not None:
            resolved_dim = getattr(src, "hidden_size", getattr(src, "embed_dim", 768))
        self.hidden_size: int = int(resolved_dim or 768)
        self.dim: int = self.hidden_size

        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or _cfg_get(config, "num_attention_heads", None)
            or _cfg_get(config, "num_heads", None)
        )
        if resolved_heads is None and src is not None:
            resolved_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", 12))
        self.num_heads: int = max(1, int(resolved_heads or 12))

        resolved_kv_heads = (
            num_kv_heads
            or kwargs.get("num_key_value_heads")
            or _cfg_get(config, "num_key_value_heads", None)
            or _cfg_get(config, "num_kv_heads", None)
            or self.num_heads
        )
        self.num_kv_heads: int = max(1, min(int(resolved_kv_heads), self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)

        resolved_head_dim = head_dim or kwargs.get("head_dim") or _cfg_get(config, "head_dim", None)
        self.head_dim: int = int(resolved_head_dim or (self.hidden_size // self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        resolved_is_decoder = (
            is_decoder
            if is_decoder is not None
            else is_causal
            if is_causal is not None
            else bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "is_causal", False)
                or kwargs.get("is_causal", False)
                or _cfg_get(config, "model_type", "") in ("gpt2", "qwen", "qwen2", "llama", "mistral")
            )
        )
        self.is_decoder: bool = resolved_is_decoder

        self.preferred_backend: SDPABackend = (
            SDPABackend(preferred_backend) if isinstance(preferred_backend, str) else preferred_backend
        )

        self.lora_rank: int = int(getattr(config, "lora_rank", lora_rank))
        self.query = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.key = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.value = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=bias,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )

        dropout_p = float(
            dropout or _cfg_get(config, "dropout", 0.0) or _cfg_get(config, "attention_probs_dropout_prob", 0.0) or 0.0
        )
        self.dropout_p: float = dropout_p
        self.scale: float = 1.0 / math.sqrt(max(int(self.head_dim), 1))

    @staticmethod
    def _read_cache(
        past_key_value: Any,  # reason: tuple or DynamicCache instance
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Extracts cached (K, V) tensors and historical sequence length."""
        if past_key_value is None:
            return None, None, 0

        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) == 0:
                return None, None, 0
            first = past_key_value[0]
            if isinstance(first, (tuple, list)):
                if layer_idx < len(past_key_value):
                    entry = past_key_value[layer_idx]
                    if len(entry) >= 2 and entry[0] is not None:
                        k, v = entry[0], entry[1]
                        return k, v, k.shape[-2]
                return None, None, 0
            k, v = past_key_value[0], past_key_value[1] if len(past_key_value) > 1 else None
            if k is not None:
                return k, v, k.shape[-2]
            return None, None, 0

        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            if layer_idx < len(past_key_value.key_cache):
                k = past_key_value.key_cache[layer_idx]
                v = past_key_value.value_cache[layer_idx] if hasattr(past_key_value, "value_cache") else None
                if k is not None:
                    return k, v, k.shape[-2]

        return None, None, 0

    @staticmethod
    def _write_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Any,  # reason: tuple or DynamicCache instance
        layer_idx: int,
    ) -> Any:  # reason: updated tuple or DynamicCache instance
        """Updates and returns key-value state for next generation step."""
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
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,  # reason: HF KV cache structure
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,  # reason: absorbs unused positional arguments
        **kwargs: Any,  # reason: absorbs layer_idx, use_cache, position_ids
    ) -> tuple[torch.Tensor, Any | None]:
        """Executes SDPA attention forward pass with automatic backend dispatch.

        Args:
            hidden_states: (batch, seq_len, dim)
            attention_mask: Optional attention mask.
            head_mask: Optional per-head mask.
            encoder_hidden_states: Optional cross-attention context.
            encoder_attention_mask: Optional mask for cross-attention.
            past_key_value: Cached past keys and values.
            output_attentions: Kept for API compatibility.
            output_hidden_states: Kept for API compatibility.
            return_dict: Kept for API compatibility.
            position_embeddings: Optional (cos, sin) tuple for rotary embeddings.
            *args: Additional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (output, present_key_value).
        """
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        is_cross = encoder_hidden_states is not None or kwargs.get("key_value_states") is not None
        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else kwargs.get("key_value_states", hidden_states)
        )

        B, N_q, _ = hidden_states.shape
        N_kv_new = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        layer_idx = int(kwargs.get("layer_idx", 0))

        past_k, past_v, _ = (None, None, 0)
        if not is_cross and past_key_value is not None:
            past_k, past_v, _ = self._read_cache(past_key_value, layer_idx)

        # 1. Projections
        Q = self.query(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_kv_new, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_kv_new, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. Rotary position embeddings
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            position_ids = kwargs.get("position_ids")
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        # 3. KV caching
        present_key_value: Any | None = None
        if not is_cross:
            if past_k is not None and past_v is not None:
                K = torch.cat([past_k.to(K.dtype), K], dim=-2)
                V = torch.cat([past_v.to(V.dtype), V], dim=-2)
            if past_key_value is not None or kwargs.get("use_cache", False):
                present_key_value = self._write_cache(K, V, past_key_value, layer_idx)
        else:
            present_key_value = past_key_value

        N_k = K.shape[-2]
        is_causal = bool((self.is_decoder or kwargs.get("is_causal", False)) and not is_cross and N_q > 1)

        # 4. GQA head expansion via repeat_interleave
        if self.kv_repeat > 1:
            K_full = K.repeat_interleave(self.kv_repeat, dim=1)
            V_full = V.repeat_interleave(self.kv_repeat, dim=1)
        else:
            K_full, V_full = K, V

        # 5. Mask preparation
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("mask")

        attn_mask = None
        if eff_mask is not None:
            if eff_mask.dim() == 4:
                attn_mask = eff_mask.to(Q.dtype)
            elif eff_mask.dim() == 2:
                attn_mask = ((eff_mask[:, None, None, :].float() - 1.0) * 1e4).to(Q.dtype)

        dropout_p = self.dropout_p if self.training else 0.0
        is_causal_flag = is_causal and attn_mask is None and N_q == N_k

        # 6. SDPA execution with backend selection
        context = sdpa_forward(
            Q,
            K_full,
            V_full,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal_flag,
            backend=self.preferred_backend,
        )

        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        out = self.out_proj(context)
        return out, present_key_value


__all__ = [
    "SDPAAttention",
]
