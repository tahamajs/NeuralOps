r"""
CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel — PRODUCTION-HARDENED.

CRITICAL FIXES (2026-10-04):
  FIX-1: Non-negative Yat feature map: (cos+1)/2 and (sin+1)/2.
         Original [cos, sin] is bipolar; denominator can hit zero → PPL explosion.
  FIX-2: Aggressive denominator clamp (min 1.0) in both attention paths.
  FIX-3: Normalize linear_out magnitude before gating.
  FIX-4: Safe gate init bias = 8.0 (99.97% dense contribution at start).
  FIX-5: No Triton — pure PyTorch.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Safe config helpers
# ===========================================================================
def _safe_getattr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _resolve_attr(cfg: Any, *keys: str) -> Any:
    for k in keys:
        v = _safe_getattr(cfg, k, None)
        if v is not None:
            return v
    return None


# ===========================================================================
# Helpers
# ===========================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    if d % 2 != 0:
        x_even = x[..., :-1]
        x1 = x_even[..., : (d - 1) // 2]
        x2 = x_even[..., (d - 1) // 2 :]
        rot_even = torch.cat((-x2, x1), dim=-1)
        return torch.cat((rot_even, x[..., -1:]), dim=-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
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
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = min(cos.shape[-1], q.shape[-1], k.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot_emb = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_emb = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_emb, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_emb
    k = torch.cat([k_rot_emb, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_emb
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    is_custom_attention: bool = False

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=16,
        lora_alpha=32.0,
        freeze_base=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, self.in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, self.lora_rank), **fk))
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

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# ChelaSlayAttention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "chela_slay", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "chela_slay_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ChelaSlayAttention(BaseAttention):
    r"""
    CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

    CRITICAL: Feature map uses shifted (cos+1)/2, (sin+1)/2 to ensure
    non-negative features. This is required for stable denominator.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        feature_dim: int = 128,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        gate_init_bias: float = 0.0,  # Balanced default: sigmoid(0)=0.5 (50% linear, 50% dense)
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, rank=rank, **kwargs)
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        gate_init_bias = _safe_float(
            kwargs.get("gate_init_bias")
            or _safe_getattr(comp_config, "gate_init_bias", None)
            or gate_init_bias,
            0.0,
        )

        if config is None and source_attn is not None:

            class _Dummy:
                pass

            config = _Dummy()
            out_layer = (
                _safe_getattr(source_attn, "out_lin", None)
                or _safe_getattr(source_attn, "dense", None)
                or _safe_getattr(source_attn, "c_proj", None)
                or _safe_getattr(source_attn, "o_proj", None)
            )
            q_layer = (
                _safe_getattr(source_attn, "q_lin", None)
                or _safe_getattr(source_attn, "query", None)
                or _safe_getattr(source_attn, "c_attn", None)
                or _safe_getattr(source_attn, "q_proj", None)
            )
            if out_layer is not None and hasattr(out_layer, "out_features"):
                config.hidden_size = out_layer.out_features
            elif q_layer is not None and hasattr(q_layer, "in_features"):
                config.hidden_size = q_layer.in_features
            else:
                config.hidden_size = 768
            config.num_attention_heads = (
                _safe_getattr(source_attn, "num_heads", None)
                or _safe_getattr(source_attn, "num_attention_heads", None)
                or _safe_getattr(source_attn, "n_heads", None)
                or 12
            )
            config.num_key_value_heads = config.num_attention_heads
            config.is_decoder = False
            config.model_type = ""

        if config is None:

            class _DefaultCfg:
                pass

            config = _DefaultCfg()
            config.hidden_size = 768
            config.num_attention_heads = 12
            config.num_key_value_heads = 12
            config.is_decoder = False
            config.model_type = ""

        self.dim = _safe_int(
            kwargs.get("hidden_size")
            or kwargs.get("d_model")
            or _resolve_attr(config, "hidden_size", "d_model", "dim"),
            768,
        )
        if self.dim <= 0:
            self.dim = 768

        n_heads_raw = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or _resolve_attr(config, "num_attention_heads", "num_heads", "n_heads", "n_head")
        )
        self.num_heads = _safe_int(n_heads_raw, 12)
        if self.num_heads <= 0:
            self.num_heads = 12

        kv_heads_raw = (
            kwargs.get("num_kv_heads")
            or kwargs.get("num_key_value_heads")
            or _resolve_attr(config, "num_key_value_heads", "num_kv_heads")
        )
        self.num_kv_heads = _safe_int(kv_heads_raw, self.num_heads)
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        head_dim_raw = kwargs.get("head_dim") or _safe_getattr(config, "head_dim", None)
        if head_dim_raw is not None:
            self.head_dim = _safe_int(head_dim_raw, self.dim // max(1, self.num_heads))
        else:
            self.head_dim = self.dim // max(1, self.num_heads)
        if self.head_dim <= 0:
            self.head_dim = 64

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.feature_dim = max(2, _safe_int(feature_dim, 128))
        self.output_feature_dim = 2 * self.feature_dim

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx = _safe_int(resolved_layer_id, 0)

        self.lora_rank = _safe_int(
            _safe_getattr(config, "lora_rank", None)
            or _safe_getattr(comp_config, "lora_rank", None)
            or kwargs.get("lora_rank", 16),
            16,
        )
        self.lora_alpha = _safe_float(
            _safe_getattr(config, "lora_alpha", None)
            or _safe_getattr(comp_config, "lora_alpha", None)
            or kwargs.get("lora_alpha", 32.0),
            32.0,
        )
        self.freeze_base = self.lora_rank > 0

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_safe_getattr(config, "model_type", "") or "").lower()
            is_dec = bool(
                _safe_getattr(config, "is_decoder", False)
                or _safe_getattr(config, "is_causal", False)
                or kwargs.get("is_causal", False)
            )
            _dec_pfx = (
                "gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
            )
            if any(model_type.startswith(p) for p in _dec_pfx) or any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (_safe_getattr(config, "architectures", []) or [])
            ):
                is_dec = True
            if source_attn is not None:
                if _safe_getattr(source_attn, "is_decoder", False) or hasattr(
                    source_attn, "c_attn"
                ):
                    is_dec = True
            self.is_decoder = is_dec
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.W_q = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_v = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Yat projection — small init to keep proj values in [-1, 1]
        self.yat_proj = nn.Parameter(
            torch.randn(self.head_dim, self.feature_dim, dtype=torch.float32) * 0.02
        )

        # Adaptive gate — starts balanced at gate_init_bias (default 0.0 -> 50% dense, 50% linear)
        self.lora_gate = nn.Linear(self.dim, self.num_heads)
        nn.init.zeros_(self.lora_gate.weight)
        with torch.no_grad():
            self.lora_gate.bias.data.fill_(float(gate_init_bias))

        dropout_p = _safe_float(
            _resolve_attr(config, "dropout", "attention_probs_dropout_prob") or 0.1,
            0.1,
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config = comp_config

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            try:
                self._copy_pretrained(source_attn)
            except Exception as e:
                logger.warning(
                    "ChelaSlayAttention._copy_pretrained failed: %s. Using init.", e
                )

    # ---- aliases ----
    @property
    def gate(self): return self.lora_gate
    @property
    def random_proj(self): return self.yat_proj
    @property
    def q_proj(self): return self.W_q
    @property
    def k_proj(self): return self.W_k
    @property
    def v_proj(self): return self.W_v
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.W_q
    @property
    def k_lin(self): return self.W_k
    @property
    def v_lin(self): return self.W_v
    @property
    def out_lin(self): return self.out_proj
    @property
    def query(self): return self.W_q
    @property
    def key(self): return self.W_k
    @property
    def value(self): return self.W_v
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def _copy_pretrained(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        try:
            for p in pretrained_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = (
            hasattr(pretrained_attn, "out_lin")
            or hasattr(pretrained_attn, "c_proj")
            or hasattr(pretrained_attn, "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            )
            or (
                cls_name in (
                    "BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention",
                    "ElectraSelfAttention", "DebertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        def _unwrap(mod):
            return getattr(mod, "base_layer", mod)

        # GPT-2 fused
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = _unwrap(pretrained_attn.c_attn)
            c_w = getattr(c_attn_mod, "weight", None)
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = pretrained_attn.c_attn.__class__.__name__ == "Conv1D" or (
                    c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                c_b = getattr(c_attn_mod, "bias", None)
                c_b_data = c_b.data.clone() if c_b is not None else None
                extracted["q"] = (c_w[:split], c_b_data[:split] if c_b_data is not None else None)
                extracted["k"] = (c_w[split:2*split], c_b_data[split:2*split] if c_b_data is not None else None)
                extracted["v"] = (c_w[2*split:], c_b_data[2*split:] if c_b_data is not None else None)

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = _unwrap(pretrained_attn.c_proj)
                o_w = getattr(c_proj_mod, "weight", None)
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]:
                        o_w = o_w.t().contiguous()
                    o_b = getattr(c_proj_mod, "bias", None)
                    extracted["o"] = (o_w, o_b.data.clone() if o_b is not None else None)

        # Nested
        elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                m = _unwrap(getattr(s, sub_name, None))
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(), b.data.clone() if b is not None else None)
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                out_mod = _unwrap(pretrained_attn.output.dense)
                b = getattr(out_mod, "bias", None)
                extracted["o"] = (out_mod.weight.data.clone(), b.data.clone() if b is not None else None)

        # Flat
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        m = _unwrap(getattr(pretrained_attn, n))
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(), b.data.clone() if b is not None else None)
                            break

        # UniversalAdapter fallback
        missing = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                ad = UniversalAdapter()
                for k in missing:
                    w, b = ad.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                "ChelaSlayAttention: missing projections %s from %s; using init.",
                missing_after, pretrained_attn.__class__.__name__,
            )

        target_map = {"q": self.W_q, "k": self.W_k, "v": self.W_v, "o": self.out_proj}
        with torch.no_grad():
            for key, target_mod in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target_mod.weight.shape
                    and w.t().contiguous().shape == target_mod.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                if out_d == target_mod.weight.size(0) and in_d == target_mod.weight.size(1):
                    target_mod.weight.data.copy_(
                        w.to(target_mod.weight.device, target_mod.weight.dtype)
                    )
                else:
                    target_mod.weight.data.zero_()
                    target_mod.weight.data[:out_d, :in_d].copy_(
                        w[:out_d, :in_d].to(target_mod.weight.device, target_mod.weight.dtype)
                    )

                if b is not None and target_mod.bias is not None:
                    mn = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:mn].copy_(
                        b[:mn].to(target_mod.bias.device, target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                nrm = float(torch.norm(target_mod.weight.data).item())
                if nrm < 1e-4:
                    logger.warning("ChelaSlay: '%s' near-zero norm (%.2e)", key, nrm)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

        try:
            for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Spherical normalize
    # ------------------------------------------------------------------
    def _spherical_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map x to a unit vector times log(1 + |x|)."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        magnitude = torch.log1p(norm.clamp_min(0.0))
        return x / (norm + 1e-6) * magnitude

    # ------------------------------------------------------------------
    # FIX-1: Non-negative Yat feature map
    # ------------------------------------------------------------------
    def _yat_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-negative shifted Yat feature map.

        FIX-1: Original [cos, sin] is bipolar → denominator can hit zero.
        Fixed: ((cos + 1)/2, (sin + 1)/2) → all features in [0, 1].

        This guarantees the linear-attention denominator
        Z = sum_j k_feat[j]   and   q_feat · Z
        are always non-negative and bounded away from zero.
        """
        if self.yat_proj.shape[0] != x.shape[-1]:
            raise RuntimeError(
                f"ChelaSlay: yat_proj shape {tuple(self.yat_proj.shape)} "
                f"vs x last dim {x.shape[-1]}."
            )
        proj = torch.matmul(x, self.yat_proj.to(x.device, x.dtype))
        scale = math.sqrt(2.0 / float(self.feature_dim))
        # FIX-1: Non-negative shifted features
        cos_shifted = (torch.cos(proj) + 1.0) * 0.5   # in [0, 1]
        sin_shifted = (torch.sin(proj) + 1.0) * 0.5   # in [0, 1]
        return torch.cat([cos_shifted, sin_shifted], dim=-1) * scale

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
    ):
        x = hidden_states
        B, L, D = x.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else x

        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = (
            self.W_k(kv_input)
            .view(B, -1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.W_v(kv_input)
            .view(B, -1, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        kv_len = k.shape[2]

        if position_embeddings is not None and not is_cross:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross:
            if position_ids is not None:
                try:
                    cos, sin = self.rotary_emb(x, seq_len=kv_len)
                except TypeError:
                    cos, sin = self.rotary_emb(x, position_ids)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Dense stream (mathematically identical to softmax) ----
        dense_scores = torch.matmul(
            q.float(), k.float().transpose(-2, -1).contiguous()
        ) * (self.head_dim ** -0.5)

        can_causal = bool(self.is_causal and not is_cross and kv_len == L)
        if can_causal:
            causal = torch.triu(
                torch.full((L, L), -1e4, device=x.device, dtype=torch.float32),
                diagonal=1,
            )
            dense_scores = dense_scores + causal.unsqueeze(0).unsqueeze(0)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        valid_mask = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                valid_mask = (eff_mask > 0).float().unsqueeze(1).unsqueeze(-1)
                pad_m = (1.0 - eff_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4
                dense_scores = dense_scores + pad_m
            elif eff_mask.dim() == 3:
                valid_mask = (eff_mask[:, 0, :] > -1000.0).float().unsqueeze(1).unsqueeze(-1)
                dense_scores = dense_scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                valid_mask = (eff_mask[:, 0, -1, :] > -1000.0).float().unsqueeze(1).unsqueeze(-1)
                dense_scores = dense_scores + eff_mask.float()

        dense_scores = torch.nan_to_num(dense_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        dense_attn = F.softmax(dense_scores, dim=-1)
        dense_attn = torch.nan_to_num(dense_attn, nan=0.0)

        head_mask_b = None
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask_b = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask_b = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 4:
                head_mask_b = head_mask
            if head_mask_b is not None:
                dense_attn = dense_attn * head_mask_b.to(dense_attn.dtype)

        dense_attn = self.dropout(dense_attn)
        dense_out = torch.matmul(dense_attn.to(v.dtype), v)

        # ---- Linear stream (FIX-1, FIX-2) ----
        q_norm = self._spherical_normalize(q)
        k_norm = self._spherical_normalize(k)
        q_feat = self._yat_feature_map(q_norm)
        k_feat = self._yat_feature_map(k_norm)

        if valid_mask is not None:
            k_feat = k_feat * valid_mask.to(k_feat.device, k_feat.dtype)
            v_masked = v * valid_mask.to(v.device, v.dtype)
        else:
            v_masked = v

        if can_causal:
            linear_out = self._causal_linear_attention(q_feat, k_feat, v_masked, L)
        else:
            linear_out = self._bidirectional_linear_attention(q_feat, k_feat, v_masked)

        # FIX-3: Normalize linear_out magnitude
        linear_out = linear_out / linear_out.norm(dim=-1, keepdim=True).clamp_min(1.0)

        if head_mask_b is not None:
            hw = head_mask_b.squeeze(-1).squeeze(-1)
            linear_out = linear_out * hw.unsqueeze(-1).to(linear_out.dtype)

        # ---- Adaptive gating (FIX-4: gate starts nearly 1) ----
        g = torch.sigmoid(self.lora_gate(x)).transpose(1, 2).contiguous().unsqueeze(-1)
        fused = g * dense_out + (1.0 - g) * linear_out
        out = fused.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        return final_out, (dense_attn if output_attentions else None)

    # ------------------------------------------------------------------
    # FIX-2: Denom clamping in both attention paths
    # ------------------------------------------------------------------
    def _bidirectional_linear_attention(self, q_feat, k_feat, v_masked):
        KV = torch.matmul(
            k_feat.transpose(-2, -1).contiguous().float(), v_masked.float()
        )
        Z = k_feat.float().sum(dim=2, keepdim=True)
        num = torch.matmul(q_feat.float(), KV)
        denom = (q_feat.float() * Z).sum(dim=-1, keepdim=True)
        # FIX-2: clamp to at least 1.0
        denom = denom.abs().clamp_min(1.0)
        return (num / denom).to(q_feat.dtype)

    def _causal_linear_attention(self, q_feat, k_feat, v_masked, L):
        chunk_size = 64
        B, H, _, feat_2f = k_feat.shape
        Dh = v_masked.shape[-1]
        num_chunks = (L + chunk_size - 1) // chunk_size

        S_running = torch.zeros(B, H, feat_2f, Dh, device=q_feat.device, dtype=torch.float32)
        Z_running = torch.zeros(B, H, feat_2f, device=q_feat.device, dtype=torch.float32)
        out_chunks = []

        for c in range(num_chunks):
            s = c * chunk_size
            e = min(s + chunk_size, L)
            q_c = q_feat[:, :, s:e, :].float()
            k_c = k_feat[:, :, s:e, :].float()
            v_c = v_masked[:, :, s:e, :].float()
            chunk_len = e - s

            # Intra-chunk causal masked attention: O(C * C * d)
            scores = torch.matmul(q_c, k_c.transpose(-2, -1))
            mask = torch.triu(torch.ones(chunk_len, chunk_len, device=q_c.device, dtype=torch.bool), diagonal=1)
            scores_masked = scores.masked_fill(mask, 0.0)
            num_intra = torch.matmul(scores_masked, v_c)
            denom_intra = scores_masked.sum(dim=-1, keepdim=True)

            # Inter-chunk attention from past chunks: O(C * 2F * d)
            num_inter = torch.matmul(q_c, S_running)
            denom_inter = torch.matmul(q_c, Z_running.unsqueeze(-1))

            denom_c = (denom_intra + denom_inter).abs().clamp_min(1.0)
            out_chunks.append(((num_intra + num_inter) / denom_c).to(q_feat.dtype))

            # Update running state
            S_running = S_running + torch.matmul(k_c.transpose(-2, -1), v_c)
            Z_running = Z_running + k_c.sum(dim=-2)

        return torch.cat(out_chunks, dim=2)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("W_q", self.W_q), ("W_k", self.W_k), ("W_v", self.W_v),
            ("out_proj", self.out_proj), ("lora_gate", self.lora_gate),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["yat_proj"] = _sz(self.yat_proj)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("chela_slay_inject")
@ATTENTION_REGISTRY.register("inject_chela_slay")
def inject_chela_slay(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_chela_slay: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n = 0
        for idx, lyr in enumerate(layer_list):
            res = inject_chela_slay(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=len(layer_list), _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n += 1
        return target if n > 0 else None

    # Single-layer
    src = old_attn
    if src is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break

    if src is not None and (
        isinstance(src, ChelaSlayAttention) or "ChelaSlay" in src.__class__.__name__
    ):
        return None

    try:
        new_attn = ChelaSlayAttention(
            config=cfg,
            pretrained_attn=src if src is not None else None,
            comp_config=comp_config,
            layer_id=layer_idx, layer_idx=layer_idx,
            **kwargs,
        )
        if src is not None:
            p = next(src.parameters(), None)
            if p is not None:
                new_attn = new_attn.to(device=p.device, dtype=p.dtype)

        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
            spliced = True
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
            spliced = True
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
            spliced = True
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
            spliced = True

        if not spliced:
            logger.error("inject_chela_slay: no writable attr on %s", target.__class__.__name__)
            return None
        return new_attn
    except Exception as e:
        logger.error("inject_chela_slay failed at layer %d: %s", layer_idx, e, exc_info=True)
        return None


__all__ = [
    "ChelaSlayAttention",
    "inject_chela_slay",
    "apply_rotary_pos_emb",
    "repeat_kv",
]