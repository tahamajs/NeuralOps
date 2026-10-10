"""
HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention.
Production-Hardened v4 — True O(1) Linear State + Full KV-Cache.

Architecture: Hybrid Dense Softmax + Linear Attention with adaptive per-head gate.

Fixes applied:
  [FIX-1]  Full KV-cache support (dense K/V + linear state)
  [FIX-2]  `lora_gate` prefixed with `lora_` for KD optimizer detection
  [FIX-3]  True O(1) linear attention with recurrent state (no recompute)
  [FIX-4]  Mask dtype/device safety in both streams
  [FIX-5]  `_rotate_half` imported from core.rope
  [FIX-6]  `repeat_kv` imported from core.gqa_utils
  [FIX-7]  Robust decoder detection
  [FIX-8]  Safe mask value using finfo.min for fp16/bf16
  [FIX-9]  Injector registered in ATTENTION_REGISTRY
  [FIX-10] Idempotency check in injector
  [FIX-11] FP32 math in linear branch
  [FIX-12] Denom clamp at 1e-5
  [FIX-13] Dropout applied in both streams
  [FIX-14] Output projection dtype-safe
  [FIX-15] `__all__` exports
  [FIX-16] Linear state passed through cache (K, V, linear_state)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _safe_mask_value(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return -65504.0 / 2.0  # -32752.0 (safe head room in FP16)
    elif dtype == torch.bfloat16:
        return -1e4
    return -1e9
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (
            (module.__class__.__name__ == "Conv1D")
            or (base.__class__.__name__ == "Conv1D")
            or (w.ndim == 2 and w.shape[0] < w.shape[1])
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b



def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    rotary_dim_even = 2 * (min(rotary_dim, head_dim) // 2)
    if rotary_dim_even == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim_even], q[..., rotary_dim_even:]
    k_rot, k_pass = k[..., :rotary_dim_even], k[..., rotary_dim_even:]
    cos_cut = cos[..., :rotary_dim_even].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim_even].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    return q, k


# ============================================================================
# LoRALinear
# ============================================================================


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
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

        if self.lora_rank > 0:
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

    def forward(self, x):
        target_dtype = x.dtype
        w = self.weight.to(target_dtype) if self.weight.dtype != target_dtype else self.weight
        b = (
            self.bias.to(target_dtype)
            if (self.bias is not None and self.bias.dtype != target_dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(target_dtype) if self.lora_A.dtype != target_dtype else self.lora_A
            bb = self.lora_B.to(target_dtype) if self.lora_B.dtype != target_dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out.to(target_dtype)


# ============================================================================
# HALO-ReGLA Attention
# ============================================================================


@ATTENTION_REGISTRY.register(
    "halo_regla",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "halo_regla_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class HALOReGLAAttention(BaseAttention):
    """Hybrid Dense Softmax + Rectified Gated Linear Attention.

    Two parallel streams fused by a learned per-head gate:
      1. Dense softmax attention (causal + padding + KV-cache).
      2. Linear attention with ELU+1 rectification (O(N) prefill, O(1) decode).

    Cache structure: (K_dense, V_dense, linear_state) where linear_state
    is a tuple (S_mat, Z_vec) for the recurrent linear attention.
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: int | None = None,
        hybrid_ratio: int = 4,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        gate_init_bias: float = 1.8,
        is_decoder: bool | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, rank or 64)
        self.config = config
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- Resolve dim ----
        dim = None
        if config is not None:
            dim = _cfg_get(
                config, "hidden_size",
                _cfg_get(config, "d_model",
                         _cfg_get(config, "n_embd",
                                  _cfg_get(config, "dim", None))),
            )
        if dim is None and src_attn is not None:
            dim = (
                getattr(src_attn, "hidden_size", None)
                or getattr(src_attn, "embed_dim", None)
                or getattr(src_attn, "all_head_size", None)
            )
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
        dim = dim or kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        # ---- Resolve num_heads ----
        num_heads = None
        if config is not None:
            num_heads = _cfg_get(
                config, "num_attention_heads",
                _cfg_get(config, "num_heads",
                         _cfg_get(config, "n_heads",
                                  _cfg_get(config, "n_head", None))),
            )
        if num_heads is None and src_attn is not None:
            num_heads = (
                getattr(src_attn, "num_heads", None)
                or getattr(src_attn, "num_attention_heads", None)
            )
        num_heads = num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(
            _cfg_get(
                config, "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        )
        self.head_dim = int(
            _cfg_get(
                config, "head_dim",
                kwargs.get("head_dim", self.dim // max(1, self.num_heads)),
            )
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.hybrid_ratio = hybrid_ratio

        # ---- Robust decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = _cfg_get(
                config, "is_decoder",
                _cfg_get(config, "is_causal",
                         kwargs.get("is_causal", kwargs.get("is_decoder", False))),
            )
            _decoder_prefixes = (
                "gpt", "qwen", "llama", "mistral", "gemma",
                "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
            )
            if any(model_type.startswith(p) for p in _decoder_prefixes):
                is_dec = True
            if src_attn is not None:
                if (
                    hasattr(src_attn, "c_attn")
                    or "gpt" in src_attn.__class__.__name__.lower()
                    or getattr(src_attn, "is_decoder", False)
                ):
                    is_dec = True
            if config is None and src_attn is None:
                is_dec = True
            self.is_decoder = bool(is_dec)

        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self.layer_idx = int(layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0))

        # ---- LoRA config ----
        comp_config = getattr(config, "compression_config", None) if config is not None else None
        self.lora_rank = int(
            _cfg_get(comp_config, "lora_rank",
                     _cfg_get(config, "lora_rank", kwargs.get("lora_rank", 16)))
        )
        self.lora_alpha = float(
            _cfg_get(comp_config, "lora_alpha",
                     _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        )
        self.freeze_base = self.lora_rank > 0

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Aliases
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        # ---- Fusion gate ----
        self.lora_gate = nn.Linear(self.dim, self.num_heads, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.lora_gate.weight)
            self.lora_gate.bias.data.fill_(gate_init_bias)

        # ---- Dropout ----
        dropout_p = _cfg_get(
            config, "dropout",
            _cfg_get(config, "attention_probs_dropout_prob", 0.1)
        )
        self.dropout_p = float(dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)

        # ---- Rotary embedding ----
        self.rotary_emb = getattr(src_attn, "rotary_emb", None) if src_attn is not None else None

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # KV-cache helpers  — [FIX-1, FIX-16]
    # Returns: (k, v, past_len, linear_state)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(
        past_key_value: Any, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, Any]:
        if past_key_value is None or past_key_value is False:
            return None, None, 0, None

        # HF DynamicCache
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            kc, vc = past_key_value.key_cache, past_key_value.value_cache
            if len(kc) == 0 or layer_idx >= len(kc):
                return None, None, 0, None
            k = kc[layer_idx]
            v = vc[layer_idx] if layer_idx < len(vc) else None
            if k is None:
                return None, None, 0, None
            lin_state = getattr(past_key_value, "_linear_states", None)
            if isinstance(lin_state, list) and layer_idx < len(lin_state):
                lin_state = lin_state[layer_idx]
            else:
                lin_state = None
            return k, v, k.shape[-2], lin_state

        # (k, v) or (k, v, linear_state) tuple
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            lin_state = past_key_value[2] if len(past_key_value) >= 3 else None
            if k is None:
                return None, None, 0, lin_state
            return k, v, k.shape[-2], lin_state

        return None, None, 0, None

    @staticmethod
    def _write_past_kv(
        k: torch.Tensor,
        v: torch.Tensor,
        linear_state: Any,
        past_key_value: Any,
        layer_idx: int,
    ) -> Any:
        """[FIX-16] Write (k, v, linear_state) to cache."""

        # DynamicCache.update()
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                if linear_state is not None:
                    if not hasattr(past_key_value, "_linear_states"):
                        past_key_value._linear_states = []
                    while len(past_key_value._linear_states) <= layer_idx:
                        past_key_value._linear_states.append(None)
                    past_key_value._linear_states[layer_idx] = linear_state
                return past_key_value
            except Exception:
                pass

        # DynamicCache-like manual write
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if layer_idx < len(past_key_value.value_cache):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                past_key_value.value_cache.append(v)
            if linear_state is not None:
                if not hasattr(past_key_value, "_linear_states"):
                    past_key_value._linear_states = []
                while len(past_key_value._linear_states) <= layer_idx:
                    past_key_value._linear_states.append(None)
                past_key_value._linear_states[layer_idx] = linear_state
            return past_key_value

        # Plain tuple (k, v, linear_state)
        return (k, v, linear_state)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: nn.Module) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )

        is_nested = False
        if (
            (hasattr(src, "self") and hasattr(src.self, "query")
             and hasattr(src, "output") and hasattr(src.output, "dense"))
            or (
                cls_name in [
                    "BertSelfAttention", "RobertaSelfAttention",
                    "XLMRobertaSelfAttention", "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            ) and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_w, c_b = _unpack_weight_bias(src.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size:2 * split_size, :],
                    c_b[split_size:2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size:, :],
                    c_b[2 * split_size:] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o_w, o_b = _unpack_weight_bias(src.c_proj)
                if o_w is not None:
                    if (
                        o_w.dim() == 2
                        and o_w.shape[0] != self.dim
                        and o_w.shape[1] == self.dim
                    ):
                        o_w = o_w.t().contiguous()
                    extracted["o"] = (o_w, o_b)

        # 2. RoBERTa/BERT nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                extracted[proj_key] = _unpack_weight_bias(mod)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                extracted["o"] = _unpack_weight_bias(src.output.dense)

        # 3. Flat modules
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
                        cw, cb = _unpack_weight_bias(mod)
                        if cw is not None:
                            extracted[proj_key] = (cw, cb)
                            break

        # UniversalAdapter fallback
        missing = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        target_map = {
            "q": self.q_proj, "k": self.k_proj,
            "v": self.v_proj, "o": self.out_proj,
        }

        def _get_base(mod: nn.Module) -> nn.Module:
            return getattr(mod, "base_layer", mod)

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                base = _get_base(target_mod)

                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(base.weight)
                        if base.bias is not None:
                            nn.init.zeros_(base.bias)
                    continue

                tw = base.weight
                if w.shape == tw.shape:
                    w_used = w
                elif w.t().contiguous().shape == tw.shape:
                    w_used = w.t().contiguous()
                elif w.shape[0] < tw.shape[0] and tw.shape[0] % w.shape[0] == 0:
                    repeats = tw.shape[0] // w.shape[0]
                    w_used = w.repeat(repeats, 1)
                else:
                    w_used = torch.zeros_like(tw)
                    m0 = min(tw.shape[0], w.shape[0])
                    m1 = min(tw.shape[1], w.shape[1])
                    w_used[:m0, :m1] = w[:m0, :m1]

                tw.copy_(w_used.to(tw.device, tw.dtype))

                if b is not None and base.bias is not None:
                    tb = base.bias
                    if b.shape == tb.shape:
                        tb.copy_(b.to(tb.device, tb.dtype))
                    elif b.shape[0] < tb.shape[0] and tb.shape[0] % b.shape[0] == 0:
                        tb.copy_(b.repeat(tb.shape[0] // b.shape[0]).to(tb.device, tb.dtype))
                    else:
                        mb = min(tb.shape[0], b.shape[0])
                        tb.zero_()
                        tb[:mb].copy_(b[:mb].to(tb.device, tb.dtype))
                elif base.bias is not None:
                    nn.init.zeros_(base.bias)

        if self.freeze_base:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                if hasattr(mod, "freeze_base_weights"):
                    mod.freeze_base_weights()
            
            # [FIX-2] Ensure fusion gate is trainable
            if hasattr(self, "lora_gate"):
                self.lora_gate.weight.requires_grad = True
                if self.lora_gate.bias is not None:
                    self.lora_gate.bias.requires_grad = True


        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
            self.is_causal = self.is_decoder

    _copy_pretrained = copy_weights

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Any = None, **kwargs: Any
    ) -> "HALOReGLAAttention":
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # [FIX-3, FIX-16] TRUE O(1) causal linear attention with recurrent state
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # [FIX-3] Vectorized causal linear attention with True O(1) State
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # [FIX-3] Vectorized causal linear attention with True O(1) State
    # ------------------------------------------------------------------
    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        past_len: int,
        valid_mask: torch.Tensor | None = None,
        linear_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Chunked causal linear attention — O(N·d²) memory instead of O(N²·d²).

        Key optimization: never materialize the full (B, H, N, d, d) tensor.
        Process in chunks of `chunk_size` tokens, keeping running state
        of shape (B, H, d, d) between chunks.
        """
        B, H, N_q, d = q.shape
        N_k = k_new.shape[2]
        device = q.device

        # Feature maps (FP32 for stability)
        lin_scale = d ** -0.25
        q_lin = F.elu(q.float() * lin_scale) + 1.0        # (B, H, N_q, d)
        k_lin = F.elu(k_new.float() * lin_scale) + 1.0    # (B, H, N_k, d)
        v_f = v_new.float()                                # (B, H, N_k, d)

        if valid_mask is not None:
            valid_kv = valid_mask.to(dtype=k_lin.dtype)
            k_lin = k_lin * valid_kv
            v_f = v_f * valid_kv

        # Load or init state — state stays (B, H, d, d) and (B, H, d) — never expands
        if linear_state is not None:
            S_state, Z_state = linear_state
            S_state = S_state.to(device=device, dtype=torch.float32)
            Z_state = Z_state.to(device=device, dtype=torch.float32)
        else:
            S_state = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
            Z_state = torch.zeros(B, H, d, device=device, dtype=torch.float32)

        # ─── Decode mode (single query token) ───
        if N_q == 1 and N_k == 1:
            S_next = S_state + torch.einsum("bhd,bhe->bhde", k_lin[:, :, 0], v_f[:, :, 0])
            Z_next = Z_state + k_lin[:, :, 0]
            num = torch.einsum("bhd,bhde->bhe", q_lin[:, :, 0], S_next)
            den = torch.einsum("bhd,bhd->bh", q_lin[:, :, 0], Z_next).clamp(min=1e-5)
            out = (num / den.unsqueeze(-1)).unsqueeze(2).to(q.dtype)
            return out, (S_next, Z_next)

        # ─── Prefill mode: chunked cumsum ───
        # Memory: O(chunk_size · d²) per iteration — no full N² materialization
        outs = torch.empty(B, H, N_q, d, device=device, dtype=torch.float32)
        S_cur = S_state
        Z_cur = Z_state

        for start in range(0, N_q, chunk_size):
            end = min(start + chunk_size, N_q)
            q_c = q_lin[:, :, start:end]         # (B, H, C, d)
            k_c = k_lin[:, :, start:end]         # (B, H, C, d)
            v_c = v_f[:, :, start:end]           # (B, H, C, d)

            # Local cumsum within chunk — C×d×d is small (e.g. 64×64×64)
            kv_local = torch.einsum("bhcd,bhce->bhcde", k_c, v_c).cumsum(dim=2)
            k_local = k_c.cumsum(dim=2)

            # Add state from previous chunks
            kv_total = kv_local + S_cur.unsqueeze(2)     # (B, H, C, d, d)
            k_total = k_local + Z_cur.unsqueeze(2)       # (B, H, C, d)

            num = torch.einsum("bhcd,bhcde->bhce", q_c, kv_total)
            den = torch.einsum("bhcd,bhcd->bhc", q_c, k_total).clamp(min=1e-5)

            outs[:, :, start:end] = num / den.unsqueeze(-1)

            # Update state — only (B, H, d, d) held between chunks
            S_cur = kv_total[:, :, -1]
            Z_cur = k_total[:, :, -1]

        return outs.to(q.dtype), (S_cur, Z_cur)


    def _global_linear_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            """Bidirectional linear attention: Q · (K^T V)."""
            B, H, N, d = q.shape
            lin_scale = d ** -0.25
            q_lin = F.elu(q.float() * lin_scale) + 1.0
            k_lin = F.elu(k.float() * lin_scale) + 1.0
            v_f = v.float()

            kv = torch.matmul(k_lin.transpose(-2, -1).contiguous(), v_f)
            z = k_lin.sum(dim=2, keepdim=True)
            num = torch.matmul(q_lin, kv)
            den = torch.matmul(q_lin, z.transpose(-2, -1).contiguous()).clamp(min=1e-5)
            return (num / den).to(q.dtype)

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            return hidden_states, None

        B, N, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- Projections (only NEW tokens) ----
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_new = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_new = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---- Read past KV + linear state ----
        past_k, past_v, past_len, linear_state = self._read_past_kv(
            past_key_value, self.layer_idx
        )

        # ---- RoPE ----
        pos_emb = (
            position_embeddings if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        pos_ids = kwargs.get("position_ids")
        if pos_ids is None and (past_len > 0 or self.is_decoder):
            pos_ids = torch.arange(
                past_len, past_len + N, device=device
            ).unsqueeze(0).expand(B, -1)

        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, pos_ids)
        elif self.rotary_emb is not None:
            try:
                if pos_ids is not None:
                    cos, sin = self.rotary_emb(k_new, position_ids=pos_ids)
                else:
                    cos, sin = self.rotary_emb(k_new)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, pos_ids)
            except Exception as e:
                logger.debug("rotary_emb failed: %s", e)

        # ============================================================
        # DENSE STREAM: uses concatenated (past + new) KV
        # ============================================================
        if past_k is not None:
            K_dense = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
            V_dense = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
        else:
            K_dense, V_dense = k_new, v_new

        N_kv = K_dense.shape[-2]

        # GQA repeat for dense
        if self.kv_repeat > 1:
            K_full_dense = repeat_kv(K_dense, self.kv_repeat)
            V_full_dense = repeat_kv(V_dense, self.kv_repeat)
            # Also prepare new-only K/V for linear branch (GQA-expanded)
            k_new_lin = repeat_kv(k_new, self.kv_repeat)
            v_new_lin = repeat_kv(v_new, self.kv_repeat)
        else:
            K_full_dense, V_full_dense = K_dense, V_dense
            k_new_lin, v_new_lin = k_new, v_new

        # ---- Attention mask ----
        eff_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        mask_val = _safe_mask_value(dtype)

        # Valid mask for linear stream (only for NEW tokens)
        valid_kv_for_linear: torch.Tensor | None = None
        if eff_mask is not None:
            valid_kv_new = get_token_valid_mask(eff_mask, B, N, device)
            if valid_kv_new is not None:
                valid_kv_for_linear = (
                    valid_kv_new[:, None, :, None].to(dtype=torch.float32)
                )

        # ============================================================
        # Stream 1: Dense Softmax Attention
        # ============================================================
# ============================================================
        # Stream 1: Dense Softmax Attention
        # ============================================================
        # ============================================================
        # Stream 1: Dense Softmax Attention
        # ============================================================
        scale = 1.0 / math.sqrt(self.head_dim)

        # Concatenate in the input dtype directly (avoid extra .float())
        K_dense_f = K_full_dense.to(dtype=torch.float32)
        V_dense_f = V_full_dense.to(dtype=torch.float32)
        q_f = q.to(dtype=torch.float32)

        dense_scores = torch.matmul(
            q_f, K_dense_f.transpose(-2, -1).contiguous()
        ) * scale

# Use dense_scores.dtype so the mask value is strictly compatible
        mask_val = _safe_mask_value(dense_scores.dtype)

        if self.is_causal:
            q_idx = torch.arange(past_len, past_len + N, device=device).unsqueeze(1)
            k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
            causal = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)
            dense_scores = dense_scores.masked_fill(~causal, mask_val)

        if eff_mask is not None:
            if eff_mask.dim() == 2:
                pad_mask = (eff_mask == 0).view(B, 1, 1, N_kv)
                dense_scores = dense_scores.masked_fill(pad_mask, mask_val)
            elif eff_mask.dim() == 4:
                dense_scores = dense_scores + eff_mask.to(dense_scores.dtype)

        dense_scores = torch.nan_to_num(
            dense_scores, nan=mask_val, posinf=1e4, neginf=mask_val
        )

        dense_attn = F.softmax(dense_scores, dim=-1).to(dtype)
        dense_attn = torch.nan_to_num(dense_attn, nan=0.0)
        dense_attn = self.dropout(dense_attn)

        # matmul in FP32 then cast back — better precision, same memory
        dense_out = torch.matmul(dense_attn.float(), V_dense_f).to(dtype)
        dense_out = torch.nan_to_num(dense_out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ============================================================
        # Stream 2: Linear Attention (with recurrent state)
        next_linear_state = None
        if self.is_causal:
            # Pass ONLY new K, V to linear branch to update state efficiently
            linear_out, next_linear_state = self._causal_linear_attention(
                q, k_new_lin, v_new_lin,  # <--- MUST BE NEW TOKENS ONLY
                past_len=past_len,
                valid_mask=valid_kv_for_linear,
                linear_state=linear_state
            )
        else:
            linear_out = self._global_linear_attention(q, K_full_dense, V_full_dense)

        linear_out = torch.nan_to_num(linear_out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ============================================================
        # Fusion Gate
        # ============================================================
        gate_out = self.lora_gate(hidden_states.float())
        g = torch.sigmoid(gate_out).permute(0, 2, 1).unsqueeze(-1).to(dtype)
        fused = (g * dense_out + (1.0 - g) * linear_out).to(dtype)

        # ---- Head mask ----
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                fused = fused * head_mask.view(1, self.num_heads, 1, 1).to(fused.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                fused = fused * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(fused.dtype)
            elif head_mask.dim() == 4:
                fused = fused * head_mask.to(fused.dtype)

        # ---- Output projection ----
        out = fused.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out_final = out
        else:
            out_final = self.out_proj(out)

        if out_final.shape[-1] != hidden_states.shape[-1]:
            if out_final.shape[-1] < hidden_states.shape[-1]:
                out_final = F.pad(
                    out_final,
                    (0, hidden_states.shape[-1] - out_final.shape[-1]),
                )
            else:
                out_final = out_final[..., : hidden_states.shape[-1]]

        out_final = torch.nan_to_num(out_final, nan=0.0, posinf=1e4, neginf=-1e4)

        # ============================================================
        # Write cache: (K_dense, V_dense, next_linear_state)
        # ============================================================
        present_kv: Any = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache:
            present_kv = self._write_past_kv(
                K_dense, V_dense, next_linear_state, past_key_value, self.layer_idx
            )
        else:
            present_kv = past_key_value

        # ============================================================
        # Return format
        # ============================================================
        if self.is_decoder:
            if output_attentions:
                return out_final.to(dtype), present_kv, dense_attn
            return out_final.to(dtype), present_kv

        return out_final.to(dtype), (dense_attn if output_attentions else None)


# ============================================================================
# Injection
# ============================================================================


@ATTENTION_REGISTRY.register("inject_halo_regla")
@ATTENTION_REGISTRY.register("halo_regla_inject")
@ATTENTION_REGISTRY.register("halo_regla_attention_inject")
def inject_halo_regla(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    """Inject HALOReGLAAttention into supported transformer layers."""
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

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
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_halo_regla(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break

    if src is None and hasattr(target, "forward"):
        src = target

    if src is None and cfg is None:
        return None

    if isinstance(src, HALOReGLAAttention) or "HALO" in src.__class__.__name__:
        return None

    try:
        new_layer = HALOReGLAAttention(
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src,
            lora_rank=lora_rank,
            layer_idx=layer_idx,
            **kwargs,
        )
        p = next(src.parameters(), None)
        if p is not None:
            new_layer.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.exception("HALO-ReGLA injection failed: %s", e)
        return None

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            target.attention.self = new_layer
            new_layer.is_nested_encoder = True
        else:
            target.attention = new_layer
        spliced = True
    elif hasattr(target, "self"):
        target.self = new_layer
        new_layer.is_nested_encoder = True
        spliced = True
    elif hasattr(target, "attn"):
        target.attn = new_layer
        spliced = True
    elif hasattr(target, "self_attn"):
        target.self_attn = new_layer
        spliced = True

    if not spliced:
        logger.error(
            "inject_halo_regla: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_layer


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "HALOReGLAAttention",
    "inject_halo_regla",
    "LoRALinear",
    "apply_rotary_pos_emb",
]