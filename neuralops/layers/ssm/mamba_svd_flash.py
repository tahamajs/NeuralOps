#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
MambaSVD-Flash: Hybrid Bidirectional SSM-FlashAttention

Fuses bidirectional selective state-space dynamics with flash attention kernels for fast sequence encoding.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.amp_utils import maybe_half
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


from .mamba_svd_flash_state import SVDS6StateSpace
from .mamba_svd_flash_weights import copy_weights, _init_ssm_from_attention
from .mamba_svd_flash_scan import _ssm_scan, _ssm_scan_from_module
from .mamba_svd_flash_injection import inject_mamba_svd_flash

class MambaSVDFlash(BaseAttention):
    """
    Hybrid SSM + MHA with per-chunk SVD and learnable per-head gating.

    State cache layout (for streaming):
        past_key_value = {
            "ssm_h": (B, D),           SSM hidden state h_{L-1}
            "attn_k": (B, H, L, Dh),   attention keys
            "attn_v": (B, H, L, Dh),   attention values
            "seq_len": int,
        }
    Or tuple form: (ssm_h, attn_k, attn_v, seq_len).
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 16,
        d_model: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        bottleneck_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        dim = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim") or d_model or hidden_size
        if dim is None:
            if config is not None:
                if isinstance(config, int):
                    dim = config
                elif isinstance(config, dict):
                    dim = config.get("hidden_size", config.get("dim", 128))
                else:
                    dim = getattr(config, "hidden_size", getattr(config, "dim", 128))
            elif source_attn is not None:
                cand = getattr(
                    source_attn,
                    "hidden_size",
                    getattr(
                        source_attn,
                        "embed_dim",
                        getattr(source_attn, "all_head_size", None),
                    ),
                )
                if cand is None and hasattr(source_attn, "q_proj"):
                    q = getattr(source_attn, "q_proj", None)
                    if q is not None and hasattr(q, "in_features"):
                        cand = q.in_features
                dim = cand or 128

        config_obj = None if isinstance(config, int) else config
        super().__init__(config_obj, rank if rank is not None else 16)

        self.d_model = int(dim or 128)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.rank = max(1, int(rank if rank is not None else 16))

        if config_obj is not None:
            self.lora_rank = int(
                getattr(
                    config_obj,
                    "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config_obj,
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
        self.freeze_base = self.lora_rank > 0

        model_type = str(getattr(config_obj, "model_type", "") or "").lower() if config_obj is not None else ""
        is_dec = False
        if config_obj is not None:
            is_dec = bool(getattr(config_obj, "is_decoder", False) or getattr(config_obj, "is_causal", False))
        if any(
            model_type.startswith(p)
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
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        n_heads = num_heads or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        if n_heads is None:
            if config_obj is not None:
                n_heads = getattr(
                    config_obj,
                    "num_attention_heads",
                    getattr(config_obj, "n_heads", 12),
                )
            elif source_attn is not None:
                n_heads = getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "num_heads", 12),
                )
            else:
                n_heads = 12
        self.num_heads = int(n_heads)
        if self.d_model % self.num_heads != 0:
            logger.warning("MambaSVDFlash: d_model %% num_heads != 0; forcing num_heads=1")
            self.num_heads = 1
        self.head_dim = self.d_model // max(1, self.num_heads)

        cfg_kv = None
        if config_obj is not None:
            cfg_kv = getattr(
                config_obj,
                "num_key_value_heads",
                getattr(config_obj, "num_kv_heads", None),
            )
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        bd = int(getattr(comp_config, "bottleneck_dim", bottleneck_dim) if comp_config else bottleneck_dim)
        self.ssm_fwd = SVDS6StateSpace(self.d_model, rank=min(16, self.rank), bottleneck_dim=bd)

        if not self.is_decoder:
            self.ssm_bwd = SVDS6StateSpace(self.d_model, rank=min(16, self.rank), bottleneck_dim=bd)
            self.bidirectional_mix = nn.Parameter(torch.zeros(self.num_heads))
        else:
            self.ssm_bwd = None
            self.bidirectional_mix = None

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
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.d_model,
            self.all_kv_head_size,
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

        self.gating = LoRALinear(
            self.d_model,
            self.num_heads,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        with torch.no_grad():
            nn.init.zeros_(self.gating.weight)
            if self.gating.bias is not None:
                nn.init.constant_(self.gating.bias, -2.0)

        dropout_p = 0.1
        if config_obj is not None:
            dropout_p = float(
                getattr(
                    config_obj,
                    "dropout",
                    getattr(config_obj, "attention_probs_dropout_prob", 0.1),
                )
            )
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        else:
            dropout_p = float(kwargs.get("dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

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

    def freeze_base_weights(self):
        for m in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.gating,
            self.ssm_fwd.down,
            self.ssm_fwd.up_u,
            self.ssm_fwd.up_v,
            self.ssm_fwd.up_sigma,
            self.ssm_fwd.up_b,
            self.ssm_fwd.up_c,
        ):
            for p in m.parameters():
                p.requires_grad = False
        if self.ssm_bwd is not None:
            for p in self.ssm_bwd.parameters():
                p.requires_grad = False
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.gating):
            for name, p in m.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

    def unfreeze_base_weights(self):
        for m in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.gating,
            self.ssm_fwd.down,
            self.ssm_fwd.up_u,
            self.ssm_fwd.up_v,
            self.ssm_fwd.up_sigma,
            self.ssm_fwd.up_b,
            self.ssm_fwd.up_c,
        ):
            for p in m.parameters():
                p.requires_grad = True
        if self.ssm_bwd is not None:
            for p in self.ssm_bwd.parameters():
                p.requires_grad = True


    copy_weights = copy_weights
    _init_ssm_from_attention = _init_ssm_from_attention
    _ssm_scan_from_module = _ssm_scan_from_module
    _ssm_scan = _ssm_scan

    @staticmethod
    def _normalize_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == 1:
                return m[0]
            if m.shape[-1] == L:
                return m[:, -1, :] if m.shape[1] == L else m[0, :, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m[:, 0, :]
            if m.shape == (B, L):
                return m

        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        return torch.ones(B, L, device=device, dtype=torch.bool)


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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("MambaSVDFlash does not support cross-attention.")

        if isinstance(hidden_states, (tuple, list)) and hidden_states:
            hidden_states = hidden_states[0]
        if isinstance(attention_mask, (tuple, list)) and attention_mask:
            attention_mask = attention_mask[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype
        device = hidden_states.device

        past_h = None
        past_attn_k = None
        past_attn_v = None
        past_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, dict):
                past_h = past_key_value.get("ssm_h", None)
                past_attn_k = past_key_value.get("attn_k", None)
                past_attn_v = past_key_value.get("attn_v", None)
                past_len = int(past_key_value.get("seq_len", 0))
            elif isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 3:
                    past_h = past_key_value[0]
                    past_attn_k = past_key_value[1]
                    past_attn_v = past_key_value[2]
                    past_len = (
                        int(past_key_value[3])
                        if len(past_key_value) > 3
                        else (past_attn_k.shape[-2] if past_attn_k is not None else 0)
                    )

        mask_bool = self._normalize_valid_mask(attention_mask, B, L, device)

        if self.is_decoder:
            ssm_out, ssm_h_next = self._ssm_scan_from_module(
                self.ssm_fwd,
                hidden_states,
                mask=mask_bool,
                h0=past_h,
            )
        else:
            ssm_fwd, _ = self._ssm_scan_from_module(
                self.ssm_fwd,
                hidden_states,
                mask=mask_bool,
                h0=None,
            )
            if self.ssm_bwd is not None:
                x_rev = torch.flip(hidden_states, dims=[1])
                mask_rev = torch.flip(mask_bool, dims=[1]) if mask_bool is not None else None
                ssm_bwd_rev, _ = self._ssm_scan_from_module(
                    self.ssm_bwd,
                    x_rev,
                    mask=mask_rev,
                    h0=None,
                )
                ssm_bwd = torch.flip(ssm_bwd_rev, dims=[1])
                w = torch.sigmoid(self.bidirectional_mix).view(1, self.num_heads, 1, 1).to(dtype_in)
                ssm_fwd_h = ssm_fwd.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                ssm_bwd_h = ssm_bwd.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                ssm_out = (w * ssm_fwd_h + (1 - w) * ssm_bwd_h).transpose(1, 2).contiguous().view(B, L, D)
                ssm_h_next = None
            else:
                ssm_out = ssm_fwd
                ssm_h_next = None

        Q = self.q_proj(hidden_states.to(self.q_proj.weight.dtype)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(hidden_states.to(self.k_proj.weight.dtype)).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V_a = self.v_proj(hidden_states.to(self.v_proj.weight.dtype)).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_ids is None:
            position_ids = kwargs.get("position_ids")

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.shape[-2] >= past_len + L:
                cos = cos.narrow(-2, past_len, L)
                sin = sin.narrow(-2, past_len, L)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        elif getattr(self, "rotary_emb", None) is not None:
            if position_ids is None:
                position_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0).expand(B, -1)
            cos = sin = None
            for call in (
                lambda: self.rotary_emb(K, seq_len=L),
                lambda: self.rotary_emb(K, position_ids),
                lambda: self.rotary_emb(position_ids),
            ):
                try:
                    cos, sin = call()
                    break
                except Exception:
                    continue
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=position_ids)

        if past_attn_k is not None:
            K = torch.cat([past_attn_k.to(K.dtype), K], dim=-2)
            V_a = torch.cat([past_attn_v.to(V_a.dtype), V_a], dim=-2)

        present_attn_k = K
        present_attn_v = V_a

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V_a = repeat_kv(V_a, self.kv_repeat)

        scale = 1.0 / math.sqrt(max(1, self.head_dim))
        attn_out: torch.Tensor | None = None
        probs: torch.Tensor | None = None

        can_sdpa = head_mask is None and not output_attentions and bool(mask_bool.all().item()) and past_len == 0
        if can_sdpa:
            try:
                _qs = maybe_half(Q)
                _ks = maybe_half(K)
                _vs = maybe_half(V_a)
                _is_causal = self.is_decoder and L > 1
                _sdpa_out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = _sdpa_out.to(dtype_in).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
            except Exception:
                attn_out = None

        if attn_out is None:
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

            if self.is_decoder and L > 1:
                kv_len = K.shape[-2]
                causal = torch.tril(
                    torch.ones(L, kv_len, device=device, dtype=torch.bool),
                    diagonal=past_len,
                )
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

            if not mask_bool.all().item():
                scores = scores.masked_fill(~mask_bool[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(V_a.dtype)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)

            if head_mask is not None:
                probs = self._apply_head_mask(head_mask, probs)

            attn_out = torch.matmul(probs, V_a).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            attn_out = torch.nan_to_num(attn_out, nan=0.0).to(dtype_in)

        gate_dtype = (
            self.gating.weight.dtype
            if hasattr(self.gating, "weight") and self.gating.weight is not None
            else dtype_in
        )
        gamma = torch.sigmoid(self.gating(hidden_states.to(gate_dtype))).to(dtype_in)
        ssm_h = ssm_out.to(dtype_in).view(B, L, self.num_heads, self.head_dim)
        attn_h = attn_out.to(dtype_in).view(B, L, self.num_heads, self.head_dim)
        g = gamma.unsqueeze(-1)
        Y = (g * ssm_h + (1 - g) * attn_h).contiguous().view(B, L, self.all_head_size)
        Y = torch.nan_to_num(Y, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = Y.to(dtype_in)
        else:
            out_dtype = (
                self.out_proj.weight.dtype
                if hasattr(self.out_proj, "weight") and self.out_proj.weight is not None
                else dtype_in
            )
            out = self.out_proj(Y.to(out_dtype)).to(dtype_in)

        present_key_value = {
            "ssm_h": (ssm_h_next if ssm_h_next is not None else torch.zeros(B, D, device=device, dtype=dtype_in)),
            "attn_k": present_attn_k,
            "attn_v": present_attn_v,
            "seq_len": past_len + L,
        }

        if output_attentions:
            attn_weights = probs if probs is not None else None
            return out, present_key_value, attn_weights
        return out, present_key_value

    def _apply_head_mask(self, head_mask: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """MSF-FIX-11: unified head_mask broadcast. probs: (B, H, Lq, Lk)."""
        hm = head_mask.to(probs.dtype)
        H = self.num_heads
        if hm.dim() == 1:
            if hm.numel() == H:
                hm = hm.view(1, H, 1, 1)
            else:
                hm = hm.view(1, -1, 1, 1)
        elif hm.dim() == 2:
            if hm.shape[-1] == H:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.shape[0] == H:
                hm = hm.t().view(1, H, 1, 1)
            else:
                hm = hm.view(1, hm.shape[-1], 1, 1)
        elif hm.dim() == 3:
            if hm.shape[1] == H:
                hm = hm.unsqueeze(2)
            elif hm.shape[2] == H:
                hm = hm.permute(0, 2, 1).unsqueeze(2)
            else:
                hm = hm.unsqueeze(1)
        return probs * hm

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
            ("out_proj", self.out_proj),
            ("gating", self.gating),
            ("ssm_fwd", self.ssm_fwd),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        if self.ssm_bwd is not None:
            out["ssm_bwd"] = sum(_sz(p) for p in self.ssm_bwd.parameters())
        out["total"] = sum(out.values())
        return out


__all__ = [
    MambaSVDFlash,
    SVDS6StateSpace,
    inject_mamba_svd_flash,
    apply_rotary_pos_emb,
    LoRALinear,
]

Mamba = MambaSVDFlash
