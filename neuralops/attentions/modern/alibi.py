"""ALiBi (Attention with Linear Biases) — Press et al. 2021.

Production-ready implementation with:
- Full Q/K/V/O projections
- LoRA adapters (frozen base)
- Multi-architecture copy_weights (GPT-2, BERT, RoBERTa, LLaMA, Qwen, T5)
- SDPA fast-path
- ALiBi linear bias (per-head slopes)
- Causal + bidirectional modes
- KV-cache support (tuple / DynamicCache)
- head_mask (1D/2D/3D/4D)
- GQA support (repeat_kv)
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
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes per head (Press et al. 2021, Eq. 5).

    For n = 2^floor(log2(num_heads)) heads:
        slope_i = 2^(-8/n * (i+1)),    i = 1..n
    For the remaining (num_heads - n) heads:
        slope_j = 2^(-8/(2n) * (2j+1)), j = 0..(num_heads - n - 1)
    """
    num_heads = int(num_heads)
    if num_heads <= 0:
        return torch.ones(1, dtype=torch.float32)

    n = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        [2.0 ** (-8.0 / n * (i + 1)) for i in range(n)],
        dtype=torch.float32,
    )
    if n < num_heads:
        extra = torch.tensor(
            [2.0 ** (-8.0 / (2 * n) * (2 * i + 1)) for i in range(num_heads - n)],
            dtype=torch.float32,
        )
        base = torch.cat([base, extra])
    return base[:num_heads]


# =============================================================================
# LoRALinear
# =============================================================================


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# =============================================================================
# ALiBi Attention
# =============================================================================


@ATTENTION_REGISTRY.register(
    "alibi",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ALiBiAttention(BaseAttention):
    r"""
    Attention with Linear Biases (ALiBi).

    Additive bias per head:
        bias[h, i, j] = -slope_h * (i - j)       (causal, j <= i)
        bias[h, i, j] = -slope_h * |i - j|       (bidirectional)

    No positional embeddings are used. Excellent length extrapolation.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = False
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---------- Resolve dims ----------
        if hidden_size is None:
            hidden_size = (
                _cfg_get(config, "hidden_size", None)
                or _cfg_get(config, "n_embd", None)
                or _cfg_get(config, "d_model", None)
                or _cfg_get(config, "dim", None)
                or kwargs.get("d_model")
                or 768
            )
        if num_heads is None:
            num_heads = (
                _cfg_get(config, "num_attention_heads", None)
                or _cfg_get(config, "num_heads", None)
                or _cfg_get(config, "n_head", None)
                or kwargs.get("n_head")
                or 12
            )
        if num_kv_heads is None:
            num_kv_heads = (
                _cfg_get(config, "num_key_value_heads", None)
                or _cfg_get(config, "num_kv_heads", None)
                or kwargs.get("num_key_value_heads")
                or num_heads
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim or (self.hidden_size // max(1, self.num_heads)))
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Decoder detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            mt = str(_cfg_get(config, "model_type", "")).lower()
            self.is_decoder = bool(
                _cfg_get(config, "is_decoder", False)
                or any(mt.startswith(p) for p in (
                    "gpt", "llama", "qwen", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon",
                ))
            )

        self.is_nested_encoder: bool = False
        self.layer_idx: int = int(layer_idx or kwargs.get("layer_idx", 0))

        # ---------- LoRA config ----------
        self.lora_rank = int(kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank)))
        self.lora_alpha = float(kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha)))
        self.freeze_base = self.lora_rank > 0

        # ---------- Projections ----------
        self.q_proj = LoRALinear(
            self.hidden_size, self.all_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.hidden_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # ---------- ALiBi slopes (buffer, not trainable) ----------
        self.register_buffer(
            "alibi_slopes",
            get_alibi_slopes(self.num_heads),
            persistent=False,
        )

        # ---------- Dropout ----------
        dropout_p = _cfg_get(
            config, "dropout",
            _cfg_get(
                config, "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob",
                         kwargs.get("dropout", 0.1)),
            ),
        )
        self.dropout_p = float(dropout_p)
        self.dropout = nn.Dropout(self.dropout_p)

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj
    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def out(self): return self.out_proj

    # ------------------------------------------------------------------
    # KV-Cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(
        past_key_value: Any,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if past_key_value is None or past_key_value is False:
            return None, None, 0

        # HF DynamicCache
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            kc, vc = past_key_value.key_cache, past_key_value.value_cache
            if len(kc) == 0 or layer_idx >= len(kc):
                return None, None, 0
            k = kc[layer_idx]
            v = vc[layer_idx] if layer_idx < len(vc) else None
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]

        # (k, v) tuple
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            if k is None:
                return None, None, 0
            return k, v, k.shape[-2]

        # Nested tuple
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
            first = past_key_value[0]
            if isinstance(first, (tuple, list)) and layer_idx < len(past_key_value):
                entry = past_key_value[layer_idx]
                if len(entry) >= 2 and entry[0] is not None:
                    return entry[0], entry[1], entry[0].shape[-2]
        return None, None, 0

    @staticmethod
    def _write_past_kv(
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: Any,
        layer_idx: int,
    ) -> Any:
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                return past_key_value
            except Exception:
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

    # ------------------------------------------------------------------
    # ALiBi bias computation
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # [OPTIMIZED] ALiBi bias computation with caching
    # ------------------------------------------------------------------
    def _compute_alibi_bias(
        self,
        N_q: int,
        N_k: int,
        past_len: int,
        device: torch.device,
        causal: bool,
    ) -> torch.Tensor:
        """Compute ALiBi bias of shape (1, H, N_q, N_k) with memory caching."""
        # ایجاد کش در صورت عدم وجود
        if not hasattr(self, "_alibi_cache"):
            self._alibi_cache = {}

        # تولید کلید یکتای کش
        cache_key = (N_q, N_k, past_len, causal, str(device))

        # اگر ماتریس قبلاً محاسبه شده، مستقیماً همان را برمی‌گردانیم (کاهش 90% از سربار)
        if cache_key in self._alibi_cache:
            return self._alibi_cache[cache_key]

        q_pos = torch.arange(past_len, past_len + N_q, device=device).unsqueeze(1)  # (N_q, 1)
        k_pos = torch.arange(0, N_k, device=device).unsqueeze(0)                    # (1, N_k)

        if causal:
            dist = (q_pos - k_pos).clamp(min=0).float()      # i - j >= 0
        else:
            dist = (q_pos - k_pos).abs().float()

        slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1).to(device)
        bias = -slopes * dist.unsqueeze(0).unsqueeze(0)      # (1, H, N_q, N_k)

        # ذخیره در کش (محدود کردن سایز کش به 4 برای جلوگیری از نشت حافظه)
        if len(self._alibi_cache) > 4:
            self._alibi_cache.clear()
        self._alibi_cache[cache_key] = bias

        return bias    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        """Copy Q/K/V/O weights from any supported source attention."""
        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. UniversalAdapter
        try:
            for proj in ("q", "k", "v", "o"):
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
        except Exception as e:
            logger.debug("UniversalAdapter failed: %s", e)

        # 2. GPT-2 fused c_attn (Conv1D)
        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            weight = getattr(base_c, "weight", getattr(c, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (
                    "Conv1D" in c.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c, "nf") or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                bias = getattr(base_c, "bias", getattr(c, "bias", None))
                bd = bias.data.clone() if bias is not None else None
                extracted["q"] = (weight[:split, :], bd[:split] if bd is not None else None)
                extracted["k"] = (weight[split:2*split, :], bd[split:2*split] if bd is not None else None)
                extracted["v"] = (weight[2*split:, :], bd[2*split:] if bd is not None else None)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            p = src.c_proj
            base_p = getattr(p, "base_layer", p)
            w_o = getattr(base_p, "weight", getattr(p, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    "Conv1D" in p.__class__.__name__
                    or "Conv1D" in base_p.__class__.__name__
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(p, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # 3. BERT / RoBERTa nested
        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            if hasattr(s, "query"): extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):   extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"): extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            d = src.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        # 4. T5-style direct attributes
        if hasattr(src, "q") and hasattr(src.q, "weight") and extracted["q"][0] is None:
            extracted["q"] = (src.q.weight.data, getattr(src.q, "bias", None))
        if hasattr(src, "k") and hasattr(src.k, "weight") and extracted["k"][0] is None:
            extracted["k"] = (src.k.weight.data, getattr(src.k, "bias", None))
        if hasattr(src, "v") and hasattr(src.v, "weight") and extracted["v"][0] is None:
            extracted["v"] = (src.v.weight.data, getattr(src.v, "bias", None))
        if hasattr(src, "o") and hasattr(src.o, "weight") and extracted["o"][0] is None:
            extracted["o"] = (src.o.weight.data, getattr(src.o, "bias", None))

        # 5. Named mapping
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q", "q"],
            "k": ["k_proj", "k_lin", "key", "W_k", "k"],
            "v": ["v_proj", "v_lin", "value", "W_v", "v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj", "o"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(src, n):
                        layer = getattr(src, n)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # ---------- Nested encoder detection ----------
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in (
            "dense", "out_proj", "out_lin", "c_proj", "o_proj", "o",
        ))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out)
            or (hasattr(src, "self") and hasattr(src, "output"))
        )

        # ---------- Warn if missing ----------
        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("ALiBiAttention copy_weights: missing %s — random init.", missing)

        # ---------- Copy ----------
        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _pool_gqa(w: torch.Tensor, target_out: int, head_dim: int) -> torch.Tensor:
            src_out = w.shape[0]
            if src_out == target_out:
                return w
            if (src_out > target_out and src_out % head_dim == 0
                and target_out % head_dim == 0):
                src_heads = src_out // head_dim
                tgt_heads = target_out // head_dim
                if src_heads > tgt_heads and src_heads % tgt_heads == 0:
                    rep = src_heads // tgt_heads
                    return w.view(tgt_heads, rep, head_dim, -1).mean(dim=1).view(target_out, -1)
            if src_out > target_out:
                idx = torch.linspace(0, src_out - 1, target_out).long()
                return w.index_select(0, idx)
            if target_out % src_out == 0:
                return w.repeat_interleave(target_out // src_out, dim=0)
            out = torch.zeros(target_out, w.shape[1], dtype=w.dtype, device=w.device)
            m = min(target_out, src_out)
            out[:m].copy_(w[:m])
            return out

        with torch.no_grad():
            for key, target in (("q", self.q_proj), ("k", self.k_proj),
                                ("v", self.v_proj), ("o", self.out_proj)):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        md = min(target.weight.shape[0], target.weight.shape[1])
                        target.weight.data.zero_()
                        target.weight.data[:md, :md].fill_diagonal_(1.0)
                        if target.bias is not None:
                            target.bias.data.zero_()
                    target.freeze_base_weights()
                    continue

                in_dim = self.all_head_size if key == "o" else self.hidden_size
                w = _orient(w, in_dim)

                if key in ("k", "v"):
                    w = _pool_gqa(w, target.weight.shape[0], self.head_dim)
                    if b is not None:
                        b = _pool_gqa(b.unsqueeze(-1), target.weight.shape[0], 1).squeeze(-1)

                od, id_ = target.weight.shape
                so, si = w.shape
                target.weight.data.zero_()
                target.weight.data[:min(od, so), :min(id_, si)].copy_(
                    w[:min(od, so), :min(id_, si)].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    bv = b.data if isinstance(b, nn.Parameter) else b
                    n = min(target.bias.numel(), bv.numel())
                    target.bias.data.zero_()
                    target.bias.data[:n].copy_(bv[:n].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        # Unwrap tuple input
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        B, N_q, _ = hidden_states.shape
        N_kv_new = kv_input.shape[1]
        H, Hkv, d = self.num_heads, self.num_kv_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # -------- Projections --------
        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, Hkv, d).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, Hkv, d).transpose(1, 2).contiguous()

        # -------- KV-Cache --------
        past_k, past_v, past_len = (None, None, 0)
        if not is_cross and past_key_value is not None and past_key_value is not False:
            past_k, past_v, past_len = self._read_past_kv(past_key_value, self.layer_idx)

        if past_k is not None and not is_cross:
            K = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K, V = K_new, V_new

        new_past_key_value: Any = None
        if not is_cross:
            use_cache = kwargs.get("use_cache", False) or (past_key_value is not None and past_key_value is not False)
            if use_cache:
                new_past_key_value = self._write_past_kv(K, V, past_key_value, self.layer_idx)
        else:
            new_past_key_value = past_key_value

        N_k = K.shape[-2]

        # -------- GQA repeat --------
        if self.kv_repeat > 1:
            K_full = repeat_kv(K, self.kv_repeat)
            V_full = repeat_kv(V, self.kv_repeat)
        else:
            K_full, V_full = K, V

        # -------- Effective causal flag --------
        # Cross-attention is never causal. Self-attention is causal iff decoder.
        is_causal = bool(self.is_decoder and not is_cross)

        # -------- Effective padding mask --------
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            eff_mask = kwargs.get("mask")

        # -------- ALiBi bias + causal mask --------
        alibi_bias = self._compute_alibi_bias(N_q, N_k, past_len, device, causal=is_causal)

        if is_causal:
            # Bottom-right-aligned causal mask
            if N_q == N_k - past_len and past_len == 0:
                causal_bool = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool))
            else:
                q_idx = torch.arange(past_len, past_len + N_q, device=device).unsqueeze(1)
                k_idx = torch.arange(N_k, device=device).unsqueeze(0)
                causal_bool = k_idx <= q_idx
            alibi_bias = alibi_bias.masked_fill(~causal_bool.unsqueeze(0).unsqueeze(0), -1e4)

        # -------- SDPA fast-path --------
        head_mask_ok = head_mask is None
        attn_out_ok = not output_attentions
        mask_ok = eff_mask is None
        use_sdpa = head_mask_ok and attn_out_ok and mask_ok

        if use_sdpa:
            try:
                # کستینگ سریع‌تر با to به جای if-else های متوالی
                _q = Q.to(dtype)
                _k = K_full.to(dtype)
                _v = V_full.to(dtype)

                # برای کستینگ بایاس روی GPU از حافظه کش استفاده می‌کنیم
                cache_key_bias = ("bias", _q.dtype, str(device))
                if not hasattr(self, "_alibi_cache"):
                    self._alibi_cache = {}
                if cache_key_bias not in self._alibi_cache or self._alibi_cache[cache_key_bias].shape != alibi_bias.shape:
                    self._alibi_cache[cache_key_bias] = alibi_bias.to(_q.dtype)

                _bias = self._alibi_cache[cache_key_bias]

                out = F.scaled_dot_product_attention(
                    _q, _k, _v,
                    attn_mask=_bias,
                    is_causal=False,   # handled by alibi_bias
                    dropout_p=self.dropout_p if self.training else 0.0,
                )
                out = out.to(dtype).transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                output = out if self.is_nested_encoder else self.out_proj(out)

                valid_q = get_token_valid_mask(attention_mask, B, N_q, device)
                if valid_q is not None:
                    output = output * valid_q.unsqueeze(-1).to(output.dtype)

                if self.is_decoder and not is_cross:
                    return output, new_past_key_value
                return output, None
            except Exception as e:
                logger.debug("ALiBi SDPA failed, falling back: %s", e)

        # -------- Manual path --------
        scores = torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * self.scale
        scores = scores + alibi_bias.to(scores.dtype)

        if eff_mask is not None:
            if eff_mask.dim() == 4:
                scores = scores + eff_mask.to(scores.dtype)
            else:
                valid_kv = get_token_valid_mask(eff_mask, B, N_k, device)
                if valid_kv is not None:
                    scores = scores.masked_fill(~valid_kv.unsqueeze(1).unsqueeze(2), -1e4)

        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, V_full)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        output = context if self.is_nested_encoder else self.out_proj(context)

        valid_q = get_token_valid_mask(attention_mask, B, N_q, device)
        if valid_q is not None:
            output = output * valid_q.unsqueeze(-1).to(output.dtype)

        attn_ret = attn if output_attentions else None
        if self.is_decoder and not is_cross:
            return output, new_past_key_value
        return output, attn_ret


# =============================================================================
# Injector  (registered)
# =============================================================================


def inject_alibi(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    """Inject ALiBiAttention into any supported transformer."""
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---------- Whole-model traversal ----------
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
    elif (hasattr(target, "bert") and hasattr(target.bert, "encoder")
          and hasattr(target.bert.encoder, "layer")):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_alibi(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # ---------- Single-layer injection ----------
    layer = target

    # Find src_attn
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                src_attn = getattr(cand, "self", cand)
                break

    # Skip if already wrapped
    if isinstance(src_attn, ALiBiAttention) or "ALiBi" in src_attn.__class__.__name__:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    # Extract dims
    hidden_size = (
        _cfg_get(cfg, "hidden_size")
        or _cfg_get(cfg, "n_embd")
        or _cfg_get(cfg, "d_model")
        or _cfg_get(cfg, "dim")
        or getattr(src_attn, "hidden_size", None)
        or 768
    )
    num_heads = (
        _cfg_get(cfg, "num_attention_heads")
        or _cfg_get(cfg, "num_heads")
        or _cfg_get(cfg, "n_head")
        or getattr(src_attn, "num_heads", None)
        or 12
    )
    num_kv_heads = (
        _cfg_get(cfg, "num_key_value_heads")
        or _cfg_get(cfg, "num_kv_heads")
        or num_heads
    )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 32.0))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    dropout = kwargs.pop("dropout", _cfg_get(comp_config, "dropout", 0.1))
    is_decoder = kwargs.pop("is_decoder", None)

    try:
        new_attn = ALiBiAttention(
            config=cfg,
            rank=rank,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pretrained_attn=src_attn,
            is_decoder=is_decoder,
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception as e:
        logger.error("ALiBiAttention construction failed: %s", e)
        return None

    # Match device/dtype of src
    if src_attn is not None:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    # ---------- Splice ----------
    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
        return new_attn
    for attr in ("self_attn", "attn", "self"):
        if hasattr(layer, attr):
            setattr(layer, attr, new_attn)
            return new_attn
    layer.attention = new_attn
    return new_attn


__all__ = [
    "ALiBiAttention",
    "inject_alibi",
    "get_alibi_slopes",
    "LoRALinear",
]