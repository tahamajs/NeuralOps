#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet Attention

Associative memory network utilizing delta update rules and gating mechanisms for constant-memory sequence modeling.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.common import _cfg_get, repeat_kv

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.ops.triton.gdn_scan import MAX_REGISTER_DIM, gdn_scan_triton

    HAS_TRITON_GDN = True
    logger.info("GDN: Triton scan available")
except Exception as _e:
    logger.warning("GDN: Triton scan unavailable: %s", _e)
    gdn_scan_triton = None
    HAS_TRITON_GDN = False
    MAX_REGISTER_DIM = 64


from ._gated_delta_net_scan import (
    HAS_TRITON_GDN,
    MAX_REGISTER_DIM,
    _clamp_fp16_safe_to_dtype,
    _delta_chunk_forward,
    chunked_delta_rule_scan,
    gdn_scan_triton,
)


@ATTENTION_REGISTRY.register(
    "gated_delta_net",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "gated_deltanet",
    is_alias=True,
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class GatedDeltaNet(BaseAttention):
    """
    Gated DeltaNet with data-dependent decay (beta) and step size (alpha).

    Recurrence:
        S_t = beta_t · S_{t-1} + step_t · k_tᵀ (v_t − k_t · S_{t-1})
        y_t = q_t · S_t
        out = y ⊙ SiLU(z)

    Supports:
        * Full KV-cache / recurrent-state passthrough
        * Bidirectional scan for encoders
        * GQA (num_kv_heads < num_heads)
        * RoPE (partial or full)
        * head_mask
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        old_attn: nn.Module | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None and config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif d_model is None and source_attn is not None:
                out_layer = None
                for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                    if hasattr(source_attn, attr):
                        out_layer = getattr(source_attn, attr)
                        break
                if out_layer is not None:
                    d_model = getattr(
                        out_layer,
                        "out_features",
                        getattr(out_layer, "in_features", 768),
                    )
                else:
                    for attr in ("q_proj", "query", "q_lin", "W_q"):
                        q_l = getattr(source_attn, attr, None)
                        if q_l is not None:
                            d_model = getattr(
                                q_l,
                                "in_features",
                                getattr(q_l, "out_features", 768),
                            )
                            break
            if d_model is None:
                d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None and config is not None:
                num_heads = getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "n_heads", getattr(config, "num_heads", 12)),
                )
            elif num_heads is None and source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "num_heads", 12),
                    ),
                )
            if num_heads is None:
                num_heads = 12

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.d_model = int(d_model)
        self.num_heads = max(1, int(num_heads))

        cfg_kv = None
        if config is not None:
            cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        head_dim = getattr(config, "head_dim", None) if config else None
        if head_dim is None and source_attn is not None:
            for q_cand in ("q_proj", "query", "q_lin", "W_q"):
                q_mod = getattr(source_attn, q_cand, None)
                if q_mod is not None and hasattr(q_mod, "out_features"):
                    head_dim = q_mod.out_features // max(1, self.num_heads)
                    break
        if head_dim is None:
            head_dim = self.d_model // self.num_heads
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        )
        self.layer_idx = int(resolved_layer_id)

        if config is not None:
            self.lora_rank = int(
                getattr(
                    config,
                    "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config,
                    "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))

        explicit_freeze = kwargs.get("freeze_base", None)
        if explicit_freeze is None and comp_config is not None:
            explicit_freeze = getattr(comp_config, "freeze_base", None)
        self.freeze_base = bool(explicit_freeze) if explicit_freeze is not None else (self.lora_rank > 0)

        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "is_causal", False))
        decoder_prefixes = (
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
            "codegen",
        )
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn"):
                is_dec = True
        if (config is None or getattr(config, "_is_synthetic", False)) and source_attn is None:
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder

        is_nested = False
        if source_attn is not None:
            src_cls = source_attn.__class__.__name__
            if (
                src_cls
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
            ) or (hasattr(source_attn, "self") and hasattr(source_attn, "output")):
                is_nested = True
        if not is_nested and model_type in ("roberta", "bert", "xlm-roberta") and not self.is_decoder:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        self.q_proj = LoRALinear(
            self.d_model,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.z_proj = LoRALinear(
            self.d_model,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.lora_output_scale = nn.Parameter(torch.ones(1))
        self.lora_beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.lora_alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        with torch.no_grad():
            nn.init.constant_(self.lora_beta_proj.bias, 2.0)
            nn.init.zeros_(self.lora_beta_proj.weight)
            nn.init.constant_(self.lora_alpha_proj.bias, 0.0)
            nn.init.zeros_(self.lora_alpha_proj.weight)

        cs = int(_cfg_get(comp_config, "chunk_size", kwargs.get("chunk_size", 32)))
        self.chunk_size = max(8, min(cs, 128))

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        self.comp_config = comp_config

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def beta_proj(self) -> nn.Linear:
        return self.lora_beta_proj

    @property
    def alpha_proj(self) -> nn.Linear:
        return self.lora_alpha_proj

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    def freeze_base_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.z_proj, self.out_proj):
            if hasattr(m, "freeze_base_weights"):
                m.freeze_base_weights()

    @property
    def W_q(self):
        return self.q_proj

    @property
    def W_k(self):
        return self.k_proj

    @property
    def W_v(self):
        return self.v_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    @staticmethod
    def _read_past_state(past_key_value: Any) -> torch.Tensor | None:
        """Extract the recurrent state from a cache object or tuple."""
        if past_key_value is None or past_key_value is False:
            return None

        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) >= 1 and isinstance(past_key_value[0], torch.Tensor):
                s = past_key_value[0]
                if s.dim() == 4 and s.shape[-1] == s.shape[-2]:
                    return s

        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            return None

        return None

    @staticmethod
    def _write_past_state(state: torch.Tensor, past_key_value: Any) -> Any:
        """Package the new state into the cache format."""
        if (
            past_key_value is not None
            and hasattr(past_key_value, "update")
            and not isinstance(past_key_value, (tuple, list))
        ):
            return (state,)
        return (state,)

    @staticmethod
    def _normalize_valid_mask(mask: torch.Tensor | None, B: int, L: int, device: torch.device) -> torch.Tensor | None:
        """Return (B, L) bool where True = valid, or None."""
        if mask is None:
            return None

        def _to_bool(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape == (B, L):
                return m
        elif mask.dim() == 3:
            m = _to_bool(mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    def copy_weights(self, old_attn: Any) -> None:
        from ._gated_delta_net_utils import copy_gdn_weights

        copy_gdn_weights(self, old_attn)

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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        past_state = None
        past_len = 0
        if not is_cross and past_key_value is not None:
            past_state = self._read_past_state(past_key_value)
            if past_state is not None:
                past_len = getattr(past_key_value, "past_len", 0)
                if past_len == 0 and isinstance(past_key_value, (tuple, list)):
                    if len(past_key_value) >= 2 and isinstance(past_key_value[1], int):
                        past_len = past_key_value[1]

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        z = self.z_proj(hidden_states)

        if not is_cross:
            if position_embeddings is not None and isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                if cos.dim() >= 3 and past_len > 0:
                    cos = cos[..., past_len : past_len + L, :]
                    sin = sin[..., past_len : past_len + L, :]
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            elif getattr(self, "rotary_emb", None) is not None:
                if position_ids is None and past_len > 0:
                    position_ids = (
                        torch.arange(past_len, past_len + kv_len, device=hidden_states.device)
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                try:
                    cos, sin = self.rotary_emb(k, seq_len=past_len + kv_len)
                    if past_len > 0:
                        cos = cos[..., past_len : past_len + L, :]
                        sin = sin[..., past_len : past_len + L, :]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                except Exception as e:
                    logger.debug("rotary_emb forward failed: %s", e)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        hs_beta = hidden_states.to(self.lora_beta_proj.weight.dtype)
        beta = torch.sigmoid(self.lora_beta_proj(hs_beta)).transpose(1, 2).contiguous()
        hs_alpha = hidden_states.to(self.lora_alpha_proj.weight.dtype)
        alpha = torch.sigmoid(self.lora_alpha_proj(hs_alpha)).transpose(1, 2).contiguous()

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        mask_bool = self._normalize_valid_mask(eff_mask, B, kv_len, hidden_states.device)

        y_fwd, present_state = chunked_delta_rule_scan(
            q,
            k,
            v,
            beta,
            alpha,
            initial_state=past_state,
            mask=mask_bool,
            chunk_size=self.chunk_size,
        )

        if not self.is_decoder:
            q_b = torch.flip(q, dims=[2])
            k_b = torch.flip(k, dims=[2])
            v_b = torch.flip(v, dims=[2])
            beta_b = torch.flip(beta, dims=[2])
            alpha_b = torch.flip(alpha, dims=[2])
            mask_b = torch.flip(mask_bool, dims=[1]) if mask_bool is not None else None
            y_bwd_f, _ = chunked_delta_rule_scan(
                q_b,
                k_b,
                v_b,
                beta_b,
                alpha_b,
                initial_state=None,
                mask=mask_b,
                chunk_size=self.chunk_size,
            )
            y_bwd = torch.flip(y_bwd_f, dims=[2])
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd

        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            y = y * hm

        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        y = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        gated_y = y.to(z.dtype) * F.silu(z)

        if self.is_nested_encoder:
            out = gated_y
        else:
            out = self.out_proj(gated_y)

        out = out * self.lora_output_scale
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        present_kv = None
        use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
        if use_cache and not is_cross:
            present_kv = self._write_past_state(present_state, past_key_value)

        if self.is_decoder and not is_cross:
            return out, present_kv

        return out, None

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for GatedDeltaNet."""
        from neuralops.ops.triton.state_update import fused_recurrent_gdn_step

        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)

        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        q = self.q_proj(x).view(B, 1, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, 1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, 1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        z = self.z_proj(x)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        q_norm = F.normalize(q, p=2, dim=-1, eps=1e-12)
        k_norm = F.normalize(k, p=2, dim=-1, eps=1e-12)

        beta = torch.sigmoid(self.lora_beta_proj(x)).view(B, 1, H).transpose(1, 2).squeeze(-1)
        alpha = torch.sigmoid(self.lora_alpha_proj(x)).view(B, 1, H).transpose(1, 2).squeeze(-1)

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)

        kv_state = state["kv_state"]

        q_t = q_norm.squeeze(2)
        k_t = k_norm.squeeze(2)
        v_t = v.squeeze(2)

        out_t, kv_state = fused_recurrent_gdn_step(q_t, k_t, v_t, beta, alpha, kv_state, scale=1.0)
        state["kv_state"] = kv_state
        state["step_idx"] = step_idx + 1

        y = out_t.unsqueeze(2).transpose(1, 2).contiguous().view(B, 1, H * d)
        gated_y = y.to(z.dtype) * F.silu(z)

        if self.is_nested_encoder:
            out = gated_y
        else:
            out = self.out_proj(gated_y)

        out = out * self.lora_output_scale
        if is_2d:
            out = out.squeeze(1)

        return out, state

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out: dict[str, int] = {}
        for name, m in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("z_proj", self.z_proj),
            ("out_proj", self.out_proj),
            ("lora_beta_proj", self.lora_beta_proj),
            ("lora_alpha_proj", self.lora_alpha_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["lora_output_scale"] = _sz(self.lora_output_scale)
        out["total"] = sum(out.values())
        return out

    def get_ortho_loss(self) -> torch.Tensor:
        try:
            return torch.tensor(0.0, device=self.out_proj.weight.device)
        except Exception:
            return torch.tensor(0.0)


from ._gated_delta_net_utils import inject_gated_delta_net

__all__ = [
    "GatedDeltaNet",
    "inject_gated_delta_net",
    "chunked_delta_rule_scan",
    "apply_rotary_pos_emb",
    "LoRALinear",
    "repeat_kv",
]
