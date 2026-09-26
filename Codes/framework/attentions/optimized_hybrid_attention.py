"""Optimized Hybrid Attention (OHA) — Production-Hardened (v2).

Fuses localized sliding-window self-attention with sequence-compressed
low-rank global attention via dynamic head-wise gated modulation and exact
pretrained weight transfer.

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Decoder detection (OHA-FIX-09)
# ===========================================================================
def _detect_decoder(config: Any, is_decoder: Optional[bool], src_attn: Optional[nn.Module]) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if src_attn is not None:
        if getattr(src_attn, "is_decoder", False) or hasattr(src_attn, "c_attn"):
            return True
        if "gpt" in src_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# RoPE (OHA-FIX-01, OHA-FIX-11)
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if cos is None or sin is None:
        return q, k
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)
    return q, k


# ===========================================================================
# LoRALinear (OHA-FIX-16)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def unfreeze_base_weights(self):
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # OHA-FIX-16: dtype-safe
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)
            ) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("oha", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("optimized_hybrid_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("oha_attention", supports_causal=True, supports_gqa=True)
class OptimizedHybridAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(self, config=None, rank=None, window_size=64, compression_rank=32,
                 use_gate=True, gate_bias=2.0, pretrained_attn=None, old_attn=None,
                 comp_config=None, is_decoder=None, **kwargs):
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = kwargs.get("head_dim") or getattr(config, "head_dim", None) or (hidden_size // max(1, num_heads))

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        self.window_size = int(getattr(config, "oha_window_size", window_size) if config else window_size)
        self.window_radius = max(1, self.window_size // 2)

        cfg_cr = getattr(comp_config, "compression_rank", None) if comp_config else None
        self.compression_rank = int(cfg_cr if cfg_cr is not None else compression_rank)
        self.use_gate = bool(use_gate)

        self.is_decoder = _detect_decoder(config, is_decoder or kwargs.get("is_causal"), source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.lora_rank = int(
            getattr(config, "lora_rank", None)
            or (getattr(comp_config, "lora_rank", None) if comp_config else None)
            or kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            getattr(config, "lora_alpha", None)
            or (getattr(comp_config, "lora_alpha", None) if comp_config else None)
            or kwargs.get("lora_alpha", 32.0)
        )
        self.freeze_base = self.lora_rank > 0

        self.query = LoRALinear(self.dim, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        self.compress_k = nn.Linear(self.dim, self.compression_rank, bias=False)
        self.compress_v = nn.Linear(self.dim, self.compression_rank, bias=False)
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)

        if self.use_gate:
            self.gate = nn.Linear(self.dim, self.num_heads)
            nn.init.constant_(self.gate.bias, gate_bias)
            nn.init.normal_(self.gate.weight, std=0.02)
        else:
            self.register_parameter("gate", None)

        dropout_p = (getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
                     if config else 0.1)
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- detection helpers ----
    @staticmethod
    def _detect_hidden_size(config, src, kwargs):
        for k in ("hidden_size", "d_model", "n_embd", "dim"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("hidden_size", "d_model", "n_embd", "dim"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("out_lin", "out_proj", "o_proj", "dense", "c_proj"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "out_features"):
                    return int(mod.out_features)
            for attr in ("q_lin", "q_proj", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    return int(mod.in_features)
        return 768

    @staticmethod
    def _detect_num_heads(config, src, kwargs, hidden_size):
        for k in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for k in ("num_heads", "num_attention_heads", "n_heads"):
                v = getattr(src, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        return max(1, hidden_size // 64)

    @staticmethod
    def _detect_num_kv_heads(config, src, kwargs, num_heads):
        for k in ("num_key_value_heads", "num_kv_heads"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_key_value_heads", "num_kv_heads"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            v = getattr(src, "num_key_value_heads", None)
            if isinstance(v, int) and v > 0:
                return v
        return num_heads

    # ---- copy_weights (OHA-FIX-15) ----
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c_attn_mod.__class__.__name__ == "Conv1D"
                           or base_c.__class__.__name__ == "Conv1D"
                           or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size, :],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)
            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (c_proj_mod.__class__.__name__ == "Conv1D"
                                 or base_o.__class__.__name__ == "Conv1D"
                                 or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(),
                                  out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"OHA.copy_weights failed: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.query, "k": self.key, "v": self.value, "o": self.out_proj}

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue
                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                # OHA-FIX-15: log transposition
                if w.shape != target_mod.weight.shape and w.t().contiguous().shape == target_mod.weight.shape:
                    logger.debug("OHA.copy_weights: transposing '%s' from %s to %s",
                                 proj_key, tuple(w.shape), tuple(target_mod.weight.shape))
                    w = w.t().contiguous()

                # OHA-FIX-15: log truncation
                if w.shape != target_mod.weight.shape:
                    logger.warning(
                        "OHA.copy_weights: '%s' shape mismatch %s vs target %s; truncating.",
                        proj_key, tuple(w.shape), tuple(target_mod.weight.shape),
                    )

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(device=target_mod.weight.device,
                                        dtype=target_mod.weight.dtype)
                )
                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning("OHA: weight '%s' has near-zero norm (%.2e)", proj_key, w_norm)

        # Sanity check (OHA-FIX-13)
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dt = self.query.weight.dtype
                dummy = torch.randn(1, 4, self.dim, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("OHA forward sanity check warning: %s", e)

    # ---- mask helper (OHA-FIX-10) ----
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, N, device):
        if attention_mask is None:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            m = attention_mask
            if m.dtype == torch.bool:
                return m.squeeze(1)
            if m.dtype.is_floating_point:
                return (m > -1e4).squeeze(1)
            return (m > 0).squeeze(1)
        if attention_mask.dim() == 4:
            m = attention_mask
            if m.dtype == torch.bool:
                v = m
            elif m.dtype.is_floating_point:
                v = m > -1e4
            else:
                v = m > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ---- forward ----
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                position_embeddings=None, past_key_value=None, layer_idx=0,
                output_attentions=False, *args, **kwargs):
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            if past_len > 0:
                cos = cos[..., past_len: past_len + N, :]
                sin = sin[..., past_len: past_len + N, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        valid_mask_2d = self._prepare_valid_mask(attention_mask, B, N, device)   # (B, N) bool
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)                   # (B,1,1,N)

        scale = 1.0 / math.sqrt(d)
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # (N, N)

        # ============================================================
        # 1. Local sliding-window attention
        # ============================================================
        if self.is_causal:
            band_mask = (diff >= 0) & (diff <= self.window_radius)
        else:
            band_mask = diff.abs() <= self.window_radius
        local_mask = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d

        local_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        local_scores = local_scores.masked_fill(~local_mask, -1e4)
        local_scores = torch.nan_to_num(local_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        local_probs = F.softmax(local_scores, dim=-1).to(dtype)
        local_probs = torch.nan_to_num(local_probs, nan=0.0)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)   # (B, H, N, d)

        # ============================================================
        # 2. Global compressed attention (OHA-FIX-03, OHA-FIX-14)
        # ============================================================
        R = self.compression_rank
        chunk_size = max(1, (N + R - 1) // R)

        # Slot layout: slot r covers [r·c, min((r+1)·c - 1, N-1)]
        slot_idx = torch.arange(R, device=device)
        chunk_starts = slot_idx * chunk_size                                  # (R,)
        chunk_ends = torch.minimum((slot_idx + 1) * chunk_size - 1,
                                    torch.tensor(N - 1, device=device))       # (R,)
        slot_valid = chunk_starts < N                                         # (R,) bool

        # comp_k_raw, comp_v_raw shape (B, N, R)
        comp_k_raw = self.compress_k(hidden_states)
        comp_v_raw = self.compress_v(hidden_states)

        # Mask padding positions to -1e4 BEFORE softmax
        pad_mask_3d = (~valid_mask_2d).unsqueeze(-1)                          # (B, N, 1)
        comp_k_raw = comp_k_raw.masked_fill(pad_mask_3d, -1e4)
        comp_v_raw = comp_v_raw.masked_fill(pad_mask_3d, -1e4)

        # OHA-FIX-03: chunk-restricted pooling.
        # comp_chunk_mask[n, r] = True iff chunk_starts[r] <= n <= chunk_ends[r]
        pos_idx = torch.arange(N, device=device)                              # (N,)
        comp_chunk_mask = (pos_idx.unsqueeze(1) >= chunk_starts.unsqueeze(0)) \
                        & (pos_idx.unsqueeze(1) <= chunk_ends.unsqueeze(0))    # (N, R)
        comp_chunk_mask = comp_chunk_mask.unsqueeze(0)                        # (1, N, R)
        comp_k_raw = comp_k_raw.masked_fill(~comp_chunk_mask, -1e4)
        comp_v_raw = comp_v_raw.masked_fill(~comp_chunk_mask, -1e4)

        comp_k_weights = F.softmax(comp_k_raw.float(), dim=1).to(dtype)       # (B, N, R)
        comp_v_weights = F.softmax(comp_v_raw.float(), dim=1).to(dtype)

        # OHA-FIX-14: zero-out invalid slots (empty chunks when N < R).
        # Note: slot_valid is (R,). Broadcasting to (1, 1, R).
        slot_valid_b = slot_valid.view(1, 1, R)
        comp_k_weights = comp_k_weights * slot_valid_b.to(comp_k_weights.dtype)
        comp_v_weights = comp_v_weights * slot_valid_b.to(comp_v_weights.dtype)

        # ck, cv: (B, H, R, d)
        ck = torch.einsum("bhnd,bnr->bhrd", K, comp_k_weights)
        cv = torch.einsum("bhnd,bnr->bhrd", V, comp_v_weights)

        # Query-global attention
        global_scores = torch.matmul(Q.float(), ck.float().transpose(-2, -1).contiguous()) * scale
        global_scores = torch.nan_to_num(global_scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        if self.is_causal:
            # OHA-FIX-03: query at i sees slot r only if the entire chunk r ends at or before i.
            # chunk_ends[r] <= i
            query_pos = torch.arange(N, device=device).unsqueeze(1)           # (N, 1)
            chunk_ends_b = chunk_ends.unsqueeze(0)                            # (1, R)
            global_causal = (query_pos >= chunk_ends_b) & slot_valid.unsqueeze(0)  # (N, R)
            global_scores = global_scores.masked_fill(
                ~global_causal.unsqueeze(0).unsqueeze(0), -1e4
            )

        global_probs = F.softmax(global_scores, dim=-1).to(dtype)
        global_probs = torch.nan_to_num(global_probs, nan=0.0)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)   # (B, H, N, d)

        # ============================================================
        # 3. Gated fusion
        # ============================================================
        if self.use_gate and self.gate is not None:
            gate = torch.sigmoid(self.gate(hidden_states))    # (B, N, H)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)         # (B, H, N, 1)
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = 0.5 * local_ctx + 0.5 * global_ctx

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        # Zero-out padded query positions
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (local_probs if output_attentions else None)


# ===========================================================================
# Injection (OHA-FIX-04)
# ===========================================================================
@ATTENTION_REGISTRY.register("oha_inject")
@ATTENTION_REGISTRY.register("inject_oha")
def inject_oha(layer=None, model=None, model_config=None, comp_config=None,
               layer_idx=0, total_layers=12, old_attn=None, _depth=0, **kwargs):
    if _depth > 8:
        logger.warning("inject_oha: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_oha(layer=lyr, model_config=cfg, comp_config=comp_config,
                             layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs)
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        src_attn = old_attn if old_attn is not None else (
            getattr(layer, "attention",
            getattr(layer, "self",
            getattr(layer, "attn", None)))
        )
        is_dec = _detect_decoder(cfg, kwargs.get("is_decoder"), src_attn)

        new_attn = OptimizedHybridAttention(
            config=cfg,
            window_size=getattr(comp_config, "window_size", 64) if comp_config else 64,
            compression_rank=getattr(comp_config, "compression_rank", 32) if comp_config else 32,
            comp_config=comp_config,
            old_attn=src_attn,
            is_decoder=is_dec,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = False
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_oha failed: %s", e, exc_info=True)
        return None


inject_optimized_hybrid_attention = inject_oha
inject_optimized_hybrid = inject_oha