#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated Hybrid Kernel Attention

Fuses factorized low-rank projections with localized kernel attention via learned gating parameters.
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
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.attention_utils import safe_mask_value as _safe_mask_value
from neuralops.core.base import BaseAttention
from neuralops.core.common import _read_past_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

from neuralops.layers.hybrid._gated_hybrid.config import _clamp_to_dtype, GatedHybridKernelConfig, AegisConfigV3
from neuralops.layers.hybrid._gated_hybrid.modules import (
    FactorisedLinear,
    StableKernelFeatureMap,
    GatingModule,
    ContextFusionGate,
)
from neuralops.layers.hybrid._gated_hybrid.weight_transfer import GatedHybridWeightTransferMixin
from neuralops.layers.hybrid._gated_hybrid.injection import inject_gated_hybrid_kernel

logger = logging.getLogger(__name__)

@ATTENTION_REGISTRY.register("gated_hybrid_kernel_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_v6", is_alias=True, supports_causal=True, supports_gqa=True)
class GatedHybridKernelAttention(GatedHybridWeightTransferMixin, BaseAttention):
    """Hybrid attention: local sliding window + global FAVOR+ linear."""

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        _pretrained_qkv: Any = None,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ):
        if not isinstance(config, GatedHybridKernelConfig):
            h_size = getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))),
            )
            if h_size is None and pretrained_attn is not None:
                for obj in [pretrained_attn, getattr(pretrained_attn, "self", None)]:
                    if obj is not None and hasattr(obj, "dense") and hasattr(obj.dense, "out_features"):
                        h_size = obj.dense.out_features
                        break
            if h_size is None:
                h_size = kwargs.get("hidden_size", 768)
            n_heads = getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))),
            ) or kwargs.get("num_heads", 12)
            h_dim = getattr(config, "head_dim", getattr(config, "d_kv", None)) or kwargs.get(
                "head_dim", h_size // max(1, n_heads)
            )
            n_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None)) or kwargs.get(
                "num_key_value_heads", n_heads
            )
            dec = self._detect_decoder(config, is_decoder, pretrained_attn)
            config = GatedHybridKernelConfig(
                hidden_size=h_size,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                window_size=getattr(config, "window_size", 128),
                svd_rank=getattr(config, "svd_rank", min(384, h_size) if h_size else 384),
                low_rank_update=getattr(config, "low_rank_update", 16),
                kernel_feature_dim=getattr(config, "kernel_feature_dim", 128),
                dropout=getattr(config, "dropout", 0.1),
                fusion_gate_bias=getattr(config, "fusion_gate_bias", 2.5),
                is_decoder=dec,
            )

        super().__init__(config)
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.h = config.num_heads
        self.num_heads = config.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.h) or self.h
        if self.h % self.num_kv_heads != 0:
            raise ValueError(f"num_heads {self.h} not divisible by num_kv_heads {self.num_kv_heads}")
        self.kv_repeat = max(1, self.h // max(1, self.num_kv_heads))
        self.d = config.head_dim
        self.head_dim = self.d
        self.all_head_size = self.h * self.d
        self.all_kv_head_size = self.num_kv_heads * self.d
        self.window = config.window_size
        self.chunk_size = config.chunk_size
        self.layer_idx = int(layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0))

        self.is_decoder = (
            bool(is_decoder)
            if is_decoder is not None
            else bool(
                getattr(config, "is_decoder", False)
                or str(getattr(config, "model_type", "")).lower()
                in (
                    "gpt2",
                    "qwen",
                    "llama",
                    "mistral",
                    "gemma",
                    "phi",
                    "opt",
                    "bloom",
                    "falcon",
                )
            )
        )
        self.is_nested_encoder = bool(kwargs.get("is_nested_encoder", False))

        self.q_proj = FactorisedLinear(
            self.hidden_size,
            self.all_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.k_proj = FactorisedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.v_proj = FactorisedLinear(
            self.hidden_size,
            self.all_kv_head_size,
            config.svd_rank,
            config.low_rank_update,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size,
            self.hidden_size,
            config.svd_rank,
            config.low_rank_update,
        )

        self.pre_norm = config.use_pre_norm
        if self.pre_norm:
            self.q_norm = nn.LayerNorm(self.d)
            self.k_norm = nn.LayerNorm(self.d)

        self.kernel_map = StableKernelFeatureMap(
            head_dim=self.d,
            num_features=config.kernel_feature_dim,
            method=config.kernel_method,
            seed=config.kernel_seed,
        )

        self.gates = GatingModule(
            config.hidden_size,
            self.h,
            self.d,
            config.use_query_gate,
            config.use_input_gate,
        )
        self.fusion = ContextFusionGate(config.hidden_size, self.h, self.d, config.fusion_gate_bias)
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @staticmethod
    def _detect_decoder(config: Any, is_decoder: bool | None, old_attn: Any) -> bool:
        if is_decoder is not None:
            return bool(is_decoder)
        if config is not None:
            if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
                return True
            mt = str(getattr(config, "model_type", "")).lower()
            if mt.startswith(
                (
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
                return True
        if old_attn is not None:
            if getattr(old_attn, "is_decoder", False):
                return True
            if hasattr(old_attn, "c_attn"):
                return True
            if "gpt" in old_attn.__class__.__name__.lower():
                return True
        return False

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
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    @staticmethod
    def _raw_valid(mask, B, N_kv, device):
        if mask is None:
            return torch.ones(B, 1, 1, N_kv, device=device, dtype=torch.bool)
        if mask.dim() == 2:
            v = mask.bool() if mask.dtype == torch.bool else (mask > -1e4 if mask.dtype.is_floating_point else mask > 0)
            return v[:, None, None, :].to(torch.bool)
        if mask.dim() == 3:
            v = mask.bool() if mask.dtype == torch.bool else (mask > -1e4 if mask.dtype.is_floating_point else mask > 0)
            return v[:, None, :, :].to(torch.bool)
        if mask.dim() == 4:
            v = mask.bool() if mask.dtype == torch.bool else (mask > -1e4 if mask.dtype.is_floating_point else mask > 0)
            return v.to(torch.bool)
        raise ValueError(f"Unsupported attention_mask dim: {mask.dim()}")

    def _prepare_valid_mask(self, attention_mask, B, N_q, N_kv, device, is_cross):
        raw = self._raw_valid(attention_mask, B, N_kv, device)
        if raw.shape[-2] == 1 and N_q != 1:
            raw = raw.expand(B, 1, N_q, N_kv)
        if self.is_decoder and not is_cross:
            past_len = N_kv - N_q
            idx_q = (past_len + torch.arange(N_q, device=device)).unsqueeze(1)
            idx_k = torch.arange(N_kv, device=device).unsqueeze(0)
            causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
            raw = raw & causal
        return raw

    _read_past_kv = staticmethod(_read_past_kv)

    @staticmethod
    def _write_past_kv(k, v, past_key_value, layer_idx):
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                return past_key_value
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if layer_idx < len(past_key_value.value_cache):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                past_key_value.value_cache.append(v)
            return past_key_value
        return (k, v)

    def _sliding_window_attention(
        self,
        Q,
        K,
        V,
        valid,
        is_cross,
        past_len,
        return_attn=False,
    ):
        """Sliding-window + causal attention with chunked query execution."""
        B, h, N, d = Q.shape
        N_k = K.shape[2]
        W = self.window
        scale = 1.0 / math.sqrt(max(int(d), 1))
        is_causal_win = self.is_decoder and not is_cross
        valid_b = valid.to(device=Q.device, dtype=torch.bool)

        if not return_attn:
            chunk_size = 64
            outputs = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                q_abs_start = past_len + start
                q_abs_end = past_len + end
                k_start = max(0, q_abs_start - W)
                k_end = min(N_k, q_abs_end if is_causal_win else q_abs_end + W)
                if k_end <= k_start:
                    outputs.append(Q.new_zeros(B, h, end - start, d))
                    continue

                Qc = Q[:, :, start:end]
                Kc = K[:, :, k_start:k_end]
                Vc = V[:, :, k_start:k_end]

                abs_q = torch.arange(q_abs_start, q_abs_end, device=Q.device).unsqueeze(1)
                abs_k = torch.arange(k_start, k_end, device=Q.device).unsqueeze(0)
                band_c = ((abs_q - abs_k).abs() <= W).unsqueeze(0).unsqueeze(0)
                if is_causal_win:
                    band_c = band_c & (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)

                v_slice = (
                    valid_b[:, :, :, k_start:k_end]
                    if valid_b.shape[-2] == 1
                    else valid_b[:, :, start:end, k_start:k_end]
                )
                combined_c = band_c & v_slice

                scores_c = torch.matmul(Qc, Kc.transpose(-2, -1)) * scale
                mask_val = _safe_mask_value(scores_c.dtype)
                scores_c = scores_c.masked_fill(~combined_c, mask_val)
                scores_c = torch.nan_to_num(scores_c, nan=mask_val, posinf=1e4, neginf=mask_val)

                any_valid_c = combined_c.any(dim=-1, keepdim=True).to(scores_c.dtype)
                attn_c = F.softmax(scores_c.float(), dim=-1).to(Q.dtype) * any_valid_c
                attn_c = self.dropout(attn_c)
                outputs.append(torch.matmul(attn_c, Vc))

            out = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
            return out, None

        abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
        abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
        dist = (abs_q - abs_k).abs()
        band = (dist <= W).unsqueeze(0).unsqueeze(0)

        if is_causal_win:
            causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
            band = band & causal

        band = band.to(device=Q.device, dtype=torch.bool)
        combined = band & valid_b

        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
        mask_val = _safe_mask_value(scores.dtype)
        scores = scores.masked_fill(~combined, mask_val)
        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

        any_valid = combined.any(dim=-1, keepdim=True).to(scores.dtype)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype) * any_valid
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        return out, attn

    def _causal_linear_attention(
        self,
        Q_feat,
        K_feat,
        V,
        valid,
        q_gate,
        in_gate,
        past_len=0,
    ):
        B, H, N, F_dim = Q_feat.shape
        D = V.shape[-1]
        dtype = Q_feat.dtype

        scale_feat = 1.0 / math.sqrt(F_dim)
        Q_feat = Q_feat.float() * scale_feat
        K_feat = K_feat.float() * scale_feat
        V_f = V.float()

        if q_gate is not None:
            Q_feat = Q_feat * q_gate.float()
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            K_feat = K_feat * in_gate.float()
            V_f = V_f * in_gate.float()

        if valid is not None:
            valid_k = valid.any(dim=-2, keepdim=True).transpose(-1, -2).float()
            K_feat = K_feat * valid_k
            V_f = V_f * valid_k

        from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK, HAS_TRITON

        if (
            HAS_TRITON
            and not FORCE_PYTORCH_FALLBACK
            and Q_feat.is_cuda
            and past_len == 0
            and F_dim <= 128
            and V_f.shape[-1] <= 128
        ):
            try:
                from neuralops.ops.triton.linear_attn import fused_svd_linear_chunk_scan_triton

                return fused_svd_linear_chunk_scan_triton(
                    _clamp_to_dtype(Q_feat, dtype),
                    _clamp_to_dtype(K_feat, dtype),
                    _clamp_to_dtype(V_f, dtype),
                    chunk_size=64,
                    eps=1e-4,
                )
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        chunk_size = 64
        state_kv = torch.zeros(B, H, F_dim, D, device=Q_feat.device, dtype=torch.float32)
        state_k = torch.zeros(B, H, F_dim, device=Q_feat.device, dtype=torch.float32)

        if past_len > 0:
            for p_start in range(0, past_len, chunk_size):
                p_end = min(p_start + chunk_size, past_len)
                Kp = K_feat[:, :, p_start:p_end].float()
                Vp = V_f[:, :, p_start:p_end].float()
                state_kv = state_kv + torch.matmul(Kp.transpose(-2, -1), Vp)
                state_k = state_k + Kp.sum(dim=2)

        outs = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            Qc = Q_feat[:, :, start:end].float()
            Kc = K_feat[:, :, past_len + start : past_len + end].float()
            Vc = V_f[:, :, past_len + start : past_len + end].float()

            intra_scores = torch.matmul(Qc, Kc.transpose(-2, -1)).tril()
            if start == 0 and past_len == 0:
                num = torch.matmul(intra_scores, Vc)
                den = intra_scores.sum(dim=-1).unsqueeze(-1).clamp(min=1e-4)
            else:
                num = torch.matmul(Qc, state_kv) + torch.matmul(intra_scores, Vc)
                den = (
                    (torch.matmul(Qc, state_k.unsqueeze(-1)).squeeze(-1) + intra_scores.sum(dim=-1))
                    .unsqueeze(-1)
                    .clamp(min=1e-4)
                )
            outs.append(torch.clamp(num / den, min=-65000.0, max=65000.0))

            if end < N:
                state_kv = state_kv + torch.matmul(Kc.transpose(-2, -1), Vc)
                state_k = state_k + Kc.sum(dim=2)

        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return _clamp_to_dtype(out, dtype)

    def _noncausal_linear_attention(self, Q, K, V, valid, q_gate, in_gate):
        """Bidirectional linear attention via associativity Q(K^T V)."""
        dtype = Q.dtype
        Qk = self.q_norm(Q) if self.pre_norm else Q
        Kk = self.k_norm(K) if self.pre_norm else K
        Q_feat = self.kernel_map(Qk).float()
        K_feat = self.kernel_map(Kk).float()
        V_f = V.float()

        if q_gate is not None:
            Q_feat = Q_feat * q_gate.float()
        if in_gate is not None and in_gate.shape[2] == K_feat.shape[2]:
            in_gate_f = in_gate.float()
            K_feat = K_feat * in_gate_f
            V_f = V_f * in_gate_f

        valid_k = valid.transpose(-1, -2).float()
        K_feat = K_feat * valid_k
        V_f = V_f * valid_k

        KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V_f)
        Z = K_feat.sum(dim=-2, keepdim=True)
        num = torch.matmul(Q_feat, KV)
        den = torch.matmul(Q_feat, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
        return _clamp_to_dtype(num / den, dtype)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_embeddings=None,
        *args,
        **kwargs,
    ):
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        Q = self._reshape(self.q_proj(hidden_states), self.h)
        K_new = self._reshape(self.k_proj(kv_input), self.num_kv_heads)
        V_new = self._reshape(self.v_proj(kv_input), self.num_kv_heads)

        past_k, past_v, past_len = self._read_past_kv(past_key_value, self.layer_idx)

        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        if pos_emb is not None and not is_cross:
            cos, sin = pos_emb
            cos = cos[..., past_len : past_len + N, :]
            sin = sin[..., past_len : past_len + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V_cat = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        present_kv = past_key_value
        use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
        if use_cache and not is_cross:
            present_kv = self._write_past_kv(K_new, V_new, past_key_value, self.layer_idx)

        N_kv = K_cat.shape[2]

        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid = self._prepare_valid_mask(eff_mask, B, N, N_kv, device, is_cross)

        q_gate, in_gate = self.gates(hidden_states, Q)

        local_out, local_attn = self._sliding_window_attention(
            Q,
            K_full,
            V_full,
            valid,
            is_cross,
            past_len=past_len,
            return_attn=bool(output_attentions),
        )

        if self.is_decoder and not is_cross:
            Q_feat = self.kernel_map(Q)
            K_feat = self.kernel_map(K_full)
            global_out = self._causal_linear_attention(
                Q_feat,
                K_feat,
                V_full,
                valid,
                q_gate,
                in_gate,
                past_len=N_kv - N,
            )
        else:
            valid_kv_only = valid.any(dim=-2, keepdim=True)
            global_out = self._noncausal_linear_attention(
                Q,
                K_full,
                V_full,
                valid_kv_only,
                q_gate,
                in_gate,
            )

        fuse_gate = self.fusion(hidden_states, local_out, global_out)
        fused = fuse_gate * local_out + (1.0 - fuse_gate) * global_out

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        fused = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_decoder and not is_cross:
            if output_attentions:
                return out.to(dtype), present_kv, local_attn
            return out.to(dtype), present_kv
        return out.to(dtype), (local_attn if output_attentions else None)

    def _reshape(self, x, num_heads):
        B, N, _ = x.shape
        return x.view(B, N, num_heads, self.d).permute(0, 2, 1, 3)

    def get_ortho_loss(self):
        return torch.zeros((), device=self.out_proj.bias.device)


AegisAttentionV3 = GatedHybridKernelAttention
AegisConfigV3 = GatedHybridKernelConfig
inject_aegis_v6 = inject_gated_hybrid_kernel
AegisAttention = GatedHybridKernelAttention
AegisAttentionV6 = GatedHybridKernelAttention

__all__ = [
    GatedHybridKernelAttention,
    GatedHybridKernelConfig,
    AegisConfigV3,
    AegisAttentionV3,
    AegisAttention,
    inject_gated_hybrid_kernel,
    inject_aegis_v6,
    FactorisedLinear,
    StableKernelFeatureMap,
    GatingModule,
    ContextFusionGate,
    apply_rotary_pos_emb,
]
