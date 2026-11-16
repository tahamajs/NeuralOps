#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
VSpec++: Variational Spectral Token-Adaptive Attention

Variational spectral attention adapting frequency-domain token projections dynamically per input sequence.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.common import _cfg_get
from neuralops.core.exceptions import IncompatibleArchitectureError
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.core.utils.rope import _rotate_half
from neuralops.surgery.director import CrossAttentionLinearBridge
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


from .vspec_components import (
    DynamicLowRankProjector,
    HeadGate,
    TokenRouter,
    VariationalSpectralGate,
    _masked_mean,
)
from .vspec_weights import copy_weights
from .vspec_injection import inject_vspec, inject_vspec_attn


@ATTENTION_REGISTRY.register("vspec", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class VSpecAttention(BaseAttention):
    r"""VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module."""

    supports_gqa: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")

        cfg_head_dim = _cfg_get(config, "head_dim", kwargs.get("head_dim"))
        if cfg_head_dim is None and src_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(src_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    num_h = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
                    cfg_head_dim = mod.weight.shape[0] // max(1, num_h)
                    break

        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
        )
        is_dec = (
            (
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or any(_mt.startswith(p) for p in _decoder_prefixes)
            )
            if config
            else False
        )
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = int(cfg_head_dim) if cfg_head_dim is not None else self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = is_dec
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(
                config,
                "hidden_size",
                kwargs.get("hidden_size", kwargs.get("dim", kwargs.get("embed_dim", kwargs.get("d_model", 768)))),
            )
            self.num_heads = _cfg_get(
                config,
                "num_attention_heads",
                kwargs.get("num_heads", kwargs.get("num_attention_heads", kwargs.get("n_heads", 12))),
            )
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = (
                int(cfg_head_dim) if cfg_head_dim is not None else (self.hidden_size // max(1, self.num_heads))
            )
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = is_dec

        for dim_key in ("hidden_size", "dim", "embed_dim", "d_model"):
            if dim_key in kwargs and kwargs[dim_key] is not None:
                self.hidden_size = int(kwargs[dim_key])
                break
        num_heads_overridden = False
        for head_key in ("num_heads", "num_attention_heads", "n_heads"):
            if head_key in kwargs and kwargs[head_key] is not None:
                self.num_heads = int(kwargs[head_key])
                num_heads_overridden = True
                break
        if "num_key_value_heads" in kwargs and kwargs["num_key_value_heads"] is not None:
            self.num_kv_heads = int(kwargs["num_key_value_heads"])
        elif "num_kv_heads" in kwargs and kwargs["num_kv_heads"] is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        elif num_heads_overridden and _cfg_get(config, "num_key_value_heads", None) is None:
            self.num_kv_heads = self.num_heads
        if "head_dim" in kwargs and kwargs["head_dim"] is not None:
            self.head_dim = int(kwargs["head_dim"])
        elif cfg_head_dim is not None:
            self.head_dim = int(cfg_head_dim)
        else:
            self.head_dim = self.hidden_size // max(1, self.num_heads)

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        self.is_causal = self.is_decoder

        self.is_nested_encoder: bool = False
        self.rank: int = rank
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and src_attn is not None:
            target_dtype = next(src_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        self.Pq = DynamicLowRankProjector(
            rank=self.rank,
            in_dim=self.hidden_size,
            out_dim=self.all_head_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.Pk = DynamicLowRankProjector(
            rank=self.rank,
            in_dim=self.hidden_size,
            out_dim=self.kv_head_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.Pv = DynamicLowRankProjector(
            rank=self.rank,
            in_dim=self.hidden_size,
            out_dim=self.kv_head_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            dtype=target_dtype,
        )
        self.Po = DynamicLowRankProjector(
            rank=self.rank,
            in_dim=self.all_head_size,
            out_dim=self.hidden_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
            dtype=target_dtype,
        )

        self.sg_q = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_k = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_v = VariationalSpectralGate(self.hidden_size, self.rank)
        self.sg_o = VariationalSpectralGate(self.all_head_size, self.rank)

        self.head_gate = HeadGate(self.hidden_size, self.num_heads)
        self.router = TokenRouter(self.hidden_size, keep_ratio=0.75)

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.use_sdpa = kwargs.get("use_sdpa", _cfg_get(config, "use_sdpa", True))
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        self.cross_attn_bridge: CrossAttentionLinearBridge | None = None

        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def query(self):
        return self.Pq

    @property
    def key(self):
        return self.Pk

    @property
    def value(self):
        return self.Pv

    @property
    def q_proj(self):
        return self.Pq

    @property
    def k_proj(self):
        return self.Pk

    @property
    def v_proj(self):
        return self.Pv

    @property
    def out_proj(self):
        return self.Po

    @property
    def o_proj(self):
        return self.Po

    @property
    def dense(self):
        return self.Po

    @property
    def c_proj(self):
        return self.Po


    copy_weights = copy_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        """Forward pass for VSpecAttention supporting self- and cross-attention.

        Args:
            hidden_states: Query token activations of shape (batch_size, seq_len, hidden_size).
            attention_mask: Optional attention mask for self-attention.
            head_mask: Optional mask over attention heads.
            encoder_hidden_states: Optional cross-sequence conditioning representations.
            encoder_attention_mask: Optional mask for cross-attention encoder states.
            past_key_value: Optional cached states for incremental generation.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output all hidden states.
            return_dict: Whether to return a dict or tuple.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (output_tensor, attention_weights) or output_tensor.

        Raises:
            IncompatibleArchitectureError: If cross-attention conditioning invariants are violated.
        """
        enc_states = encoder_hidden_states if encoder_hidden_states is not None else kwargs.get("encoder_hidden_states")
        if enc_states is not None:
            if not isinstance(enc_states, torch.Tensor):
                raise IncompatibleArchitectureError(
                    f"encoder_hidden_states must be a torch.Tensor, got {type(enc_states).__name__}"
                )
            if enc_states.dim() != 3:
                raise IncompatibleArchitectureError(
                    f"encoder_hidden_states must have 3 dimensions (B, L_enc, D), got shape {enc_states.shape}"
                )
            if self.cross_attn_bridge is None:
                self.cross_attn_bridge = CrossAttentionLinearBridge(
                    head_dim=self.head_dim,
                    persistent_cache=True,
                ).to(device=hidden_states.device, dtype=hidden_states.dtype)

            B, N, D = hidden_states.shape
            enc_mask = (
                encoder_attention_mask
                if encoder_attention_mask is not None
                else kwargs.get("encoder_attention_mask")
            )
            # Route through cross_attn_bridge
            Q = self.Pq(hidden_states, None)
            K_enc = enc_states
            V_enc = enc_states
            cross_out = self.cross_attn_bridge(
                Q,
                k=K_enc,
                v=V_enc,
                encoder_hidden_states=enc_states,
                attention_mask=enc_mask,
            )
            if cross_out.shape[-1] != D:
                cross_out = F.pad(cross_out, (0, D - cross_out.shape[-1])) if cross_out.shape[-1] < D else cross_out[..., :D]
            if self.is_nested_encoder:
                return (cross_out, None)
            return (cross_out, None)

        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        valid = get_token_valid_mask(mask, B, N, device) if mask is not None else None

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "causal", False)
            or any(
                str(getattr(self.config, "model_type", "") or "").lower().startswith(p)
                for p in (
                    "gpt",
                    "qwen",
                    "llama",
                    "mistral",
                    "gemma",
                    "phi",
                    "opt",
                    "bloom",
                    "falcon",
                    "olmo",
                    "mpt",
                    "stablelm",
                )
            )
        )

        if is_decoder or self.is_nested_encoder:
            routed = hidden_states
            idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            k_len = N
            valid_routed = valid
        else:
            routed, idx = self.router(hidden_states, valid, is_decoder=False)
            k_len = routed.shape[1]
            valid_routed = torch.gather(valid, 1, idx) if valid is not None else None

        gq, _, _ = self.sg_q(routed, valid_routed)
        gk, _, _ = self.sg_k(routed, valid_routed)
        gv, _, _ = self.sg_v(routed, valid_routed)

        Q = self.Pq(routed, gq)
        K = self.Pk(routed, gk)
        V = self.Pv(routed, gv)

        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        Q = Q.view(B, k_len, H, Dh).transpose(1, 2).contiguous()
        K = K.view(B, k_len, H_kv, Dh).transpose(1, 2).contiguous()
        V = V.view(B, k_len, H_kv, Dh).transpose(1, 2).contiguous()

        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            cos_b = cos.expand(B, -1, -1) if cos.size(0) == 1 else cos
            sin_b = sin.expand(B, -1, -1) if sin.size(0) == 1 else sin
            cos_routed = torch.gather(cos_b, 1, idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
            sin_routed = torch.gather(sin_b, 1, idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))
            Q, K = apply_rotary_pos_emb(Q, K, cos_routed, sin_routed)
        elif getattr(self, "rotary_emb", None) is not None:
            try:
                from neuralops.utils.compat import maybe_apply_rotary

                Q, K = maybe_apply_rotary(
                    Q,
                    K,
                    getattr(self, "adapter", None),
                    self.rotary_emb,
                    kwargs.get("position_ids"),
                )
            except (RuntimeError, TypeError, ValueError) as e:
                from neuralops.utils.logging_config import warn_once

                warn_once("vspec_rotary_failed", "maybe_apply_rotary failed: %s", e)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        hg = self.head_gate(routed, valid_routed).unsqueeze(-1).unsqueeze(-1)

        can_use_sdpa = (
            getattr(self, "use_sdpa", True)
            and device.type == "cuda"
            and Dh <= 256
            and dtype in (torch.float16, torch.bfloat16)
            and head_mask is None
            and not output_attentions
        )

        if can_use_sdpa:
            if valid_routed is None:
                combined_mask = None
                is_causal_sdpa = bool(is_decoder and k_len > 1)
            else:
                is_causal_sdpa = False
                combined_mask = valid_routed.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, k_len, k_len)
                if is_decoder and k_len > 1:
                    causal_bool = torch.tril(torch.ones(k_len, k_len, device=device, dtype=torch.bool))
                    combined_mask = combined_mask & causal_bool.unsqueeze(0).unsqueeze(1)

            context = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=combined_mask,
                is_causal=is_causal_sdpa,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0 / math.sqrt(max(Dh, 1)),
            )
            context = context * hg.to(context.dtype)
            attn_weights = None
        else:
            scale = 1.0 / math.sqrt(max(Dh, 1))
            scores = torch.matmul(Q.float(), K.float().transpose(-1, -2).contiguous()) * scale * hg.float()

            if is_decoder and k_len > 1:
                causal_mask = torch.triu(
                    torch.full((k_len, k_len), float("-inf"), device=device, dtype=scores.dtype),
                    diagonal=1,
                )
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if valid_routed is not None:
                pad_mask = (~valid_routed).unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, k_len, k_len)
                scores = scores.masked_fill(pad_mask, float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0).to(dtype)
            attn_weights = self.dropout(attn_weights)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                elif head_mask.dim() == 4:
                    if head_mask.shape[-1] == head_mask.shape[-2] and head_mask.shape[-1] != attn_weights.shape[-1]:
                        head_mask = head_mask.mean(dim=-1, keepdim=True)
                attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

            context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(B, k_len, self.all_head_size)

        if self.is_nested_encoder:
            return (context, (attn_weights if output_attentions else None))

        go, _, _ = self.sg_o(context, valid_routed)
        out = self.Po(context, go)

        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        if k_len == N:
            output = out
        else:
            zeros = torch.zeros(B, N, D, device=device, dtype=out.dtype)
            output = zeros.scatter(1, idx.unsqueeze(-1).expand(-1, -1, D), out)

        return (output, (attn_weights if output_attentions else None))




inject_vspec = inject_vspec_attn

__all__ = [
    VSpecAttention,
    VariationalSpectralGate,
    HeadGate,
    TokenRouter,
    DynamicLowRankProjector,
    inject_vspec_attn,
    inject_vspec,
]
