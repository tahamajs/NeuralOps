"""
Weighted Grouped-Query Attention (WGQA).

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Features:
- Learnable per-KV-head importance weights
- Robust GQA pooling (mean)
- LoRA adapters
- RoPE support with fallback
- Dynamic dimension inference
- Bottom-right aligned causal masking
- Full KV-cache support (tuple / DynamicCache)
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
from neuralops.core.rope import _rotate_half
from neuralops.utils.model_family import ModelAdapter

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


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. Handles even/odd dims safely."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.dtype)
    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    return q, k


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
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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
            nn.init.zeros_(self.lora_B)
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
# WGQA
# =============================================================================


@ATTENTION_REGISTRY.register(
    "wgqa",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "wgqa_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class WeightedGroupedQueryAttention(BaseAttention):
    r"""
    Weighted Grouped-Query Attention (WGQA).
    Learnable per-KV-head importance weights + LoRA + KV-cache + causal masking.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        num_kv_heads: int | None = None,
        is_decoder: bool | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        if config is not None and not isinstance(config, int):
            self.adapter = ModelAdapter(config)
            self.num_heads: int = self.adapter.num_heads
            self.num_kv_heads: int = self.adapter.num_key_value_heads
            self.hidden_size: int = self.adapter.hidden_size
            self.head_dim: int = self.adapter.head_dim
            self.all_head_size: int = self.num_heads * self.head_dim
            self.kv_head_size: int = self.num_kv_heads * self.head_dim
            self.kv_repeat: int = self.adapter.kv_repeat
            self.is_decoder: bool = getattr(config, "is_decoder", False)
        else:
            self.adapter = None
            self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
            self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_heads", 12))
            self.num_kv_heads = _cfg_get(config, "num_key_value_heads", self.num_heads)
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            self.is_decoder = bool(_cfg_get(config, "is_decoder", kwargs.get("is_decoder", False)))

        # Prioritize explicit kwargs
        if "hidden_size" in kwargs:
            self.hidden_size = kwargs["hidden_size"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
        if "num_heads" in kwargs:
            self.num_heads = kwargs["num_heads"]
            self.head_dim = self.hidden_size // max(1, self.num_heads)
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if num_kv_heads is not None:
            self.num_kv_heads = int(num_kv_heads)
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)

        self.is_nested_encoder: bool = False
        self.lora_rank = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # Learnable per-KV-head importance weights
        self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))
        self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))

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
        self.rotary_emb = getattr(src_attn, "rotary_emb", None)

        # Layer index for cache (set by injector)
        self.layer_idx: int = int(kwargs.get("layer_idx", 0))

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def out(self): return self.out_proj
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

    # ------------------------------------------------------------------
    # KV-Cache helpers  (FIX 2)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(
        past_key_value: Any,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Read past K/V from tuple, list, or DynamicCache. Returns (k, v, past_len)."""
        if past_key_value is None or past_key_value is False:
            return None, None, 0

        # HF DynamicCache
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            kc = past_key_value.key_cache
            vc = past_key_value.value_cache
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

        # Nested tuple: ((k0, v0), (k1, v1), ...)
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
        """Write updated K/V into cache. Returns updated cache."""
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
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 c_attn Conv1D
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_mod = src.c_attn
            base_c = getattr(c_mod, "base_layer", c_mod)
            weight = getattr(base_c, "weight", getattr(c_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    "Conv1D" in c_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                extracted["q"] = (weight[:split, :], bias_data[:split] if bias_data is not None else None)
                extracted["k"] = (weight[split:2*split, :], bias_data[split:2*split] if bias_data is not None else None)
                extracted["v"] = (weight[2*split:, :], bias_data[2*split:] if bias_data is not None else None)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            p_mod = src.c_proj
            base_o = getattr(p_mod, "base_layer", p_mod)
            w_o = getattr(base_o, "weight", getattr(p_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    "Conv1D" in p_mod.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(p_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(p_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # 3. Nested BERT/RoBERTa
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query"): extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):   extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"): extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            d = src.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        # 4. Named mappings
        named_map = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(src, n):
                        layer = getattr(src, n)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Nested encoder detection
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        is_nested = False
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention",
                       "ElectraSelfAttention", "DebertaSelfAttention") or (hasattr(src, "query") and not has_out):
            is_nested = True
        if hasattr(src, "self") and hasattr(src, "output"):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("WGQA copy_weights: missing %s — proceeding with random init.", missing)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            # Q
            w_q, b_q = extracted["q"]
            if w_q is not None:
                w_q = _orient(w_q, self.hidden_size)
                od, id_ = self.q_proj.weight.shape
                so, si = w_q.shape
                self.q_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_q[:min(od,so), :min(id_,si)])
                if b_q is not None and self.q_proj.bias is not None:
                    bv = b_q.data if isinstance(b_q, nn.Parameter) else b_q
                    n = min(len(self.q_proj.bias), len(bv))
                    self.q_proj.bias.data[:n].copy_(bv[:n])

            # K with mean pooling if needed
            w_k, b_k = extracted["k"]
            if w_k is not None:
                w_k = _orient(w_k, self.hidden_size)
                if w_k.shape[0] == self.kv_head_size:
                    od, id_ = self.k_proj.weight.shape
                    so, si = w_k.shape
                    self.k_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_k[:min(od,so), :min(id_,si)])
                    if b_k is not None and self.k_proj.bias is not None:
                        bv = b_k.data if isinstance(b_k, nn.Parameter) else b_k
                        n = min(len(self.k_proj.bias), len(bv))
                        self.k_proj.bias.data[:n].copy_(bv[:n])
                else:
                    rep = max(1, w_k.shape[0] // max(1, self.kv_head_size))
                    w_k_p = w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1).view(self.kv_head_size, -1)
                    od, id_ = self.k_proj.weight.shape
                    so, si = w_k_p.shape
                    self.k_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_k_p[:min(od,so), :min(id_,si)])
                    if b_k is not None and self.k_proj.bias is not None:
                        bv = b_k.data if isinstance(b_k, nn.Parameter) else b_k
                        b_k_p = bv.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                        n = min(len(self.k_proj.bias), len(b_k_p))
                        self.k_proj.bias.data[:n].copy_(b_k_p[:n])

            # V with mean pooling
            w_v, b_v = extracted["v"]
            if w_v is not None:
                w_v = _orient(w_v, self.hidden_size)
                if w_v.shape[0] == self.kv_head_size:
                    od, id_ = self.v_proj.weight.shape
                    so, si = w_v.shape
                    self.v_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_v[:min(od,so), :min(id_,si)])
                    if b_v is not None and self.v_proj.bias is not None:
                        bv = b_v.data if isinstance(b_v, nn.Parameter) else b_v
                        n = min(len(self.v_proj.bias), len(bv))
                        self.v_proj.bias.data[:n].copy_(bv[:n])
                else:
                    rep = max(1, w_v.shape[0] // max(1, self.kv_head_size))
                    w_v_p = w_v.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1).view(self.kv_head_size, -1)
                    od, id_ = self.v_proj.weight.shape
                    so, si = w_v_p.shape
                    self.v_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_v_p[:min(od,so), :min(id_,si)])
                    if b_v is not None and self.v_proj.bias is not None:
                        bv = b_v.data if isinstance(b_v, nn.Parameter) else b_v
                        b_v_p = bv.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                        n = min(len(self.v_proj.bias), len(b_v_p))
                        self.v_proj.bias.data[:n].copy_(b_v_p[:n])

            # O
            w_o, b_o = extracted["o"]
            if w_o is not None and not self.is_nested_encoder:
                w_o = _orient(w_o, self.all_head_size)
                od, id_ = self.out_proj.weight.shape
                so, si = w_o.shape
                self.out_proj.weight.data[:min(od,so), :min(id_,si)].copy_(w_o[:min(od,so), :min(id_,si)])
                if b_o is not None and self.out_proj.bias is not None:
                    bv = b_o.data if isinstance(b_o, nn.Parameter) else b_o
                    n = min(len(self.out_proj.bias), len(bv))
                    self.out_proj.bias.data[:n].copy_(bv[:n])
            elif self.is_nested_encoder:
                md = min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:md, :md].fill_diagonal_(1.0)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

            for t in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                t.freeze_base_weights()

        # Preserve source LoRA if present
        source_submodules = {
            "q": ["q_proj", "q_lin", "query", "W_q"],
            "k": ["k_proj", "k_lin", "key", "W_k"],
            "v": ["v_proj", "v_lin", "value", "W_v"],
            "o": ["o_proj", "out_proj", "out_lin", "dense", "out", "c_proj"],
        }
        for proj_key, target in [("q", self.q_proj), ("k", self.k_proj),
                                  ("v", self.v_proj), ("o", self.out_proj)]:
            for name in source_submodules[proj_key]:
                if hasattr(src, name):
                    sub = getattr(src, name)
                    if (hasattr(sub, "lora_A") and hasattr(sub, "lora_B")
                        and sub.lora_A is not None and sub.lora_B is not None):
                        src_lA = getattr(sub.lora_A, "weight", sub.lora_A)
                        src_lB = getattr(sub.lora_B, "weight", sub.lora_B)
                        if hasattr(src_lA, "data") and target.lora_A is not None:
                            if src_lA.shape == target.lora_A.shape:
                                target.lora_A.data.copy_(src_lA.data)
                                target.lora_A.requires_grad = True
                        if hasattr(src_lB, "data") and target.lora_B is not None:
                            if src_lB.shape == target.lora_B.shape:
                                target.lora_B.data.copy_(src_lB.data)
                                target.lora_B.requires_grad = True
                        if hasattr(sub, "lora_scaling"):
                            target.lora_scaling = sub.lora_scaling
                        break

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
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k_new = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # -------- Projections --------
        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_k_new, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_k_new, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # -------- Learnable per-KV-head importance weights --------
        if not hasattr(self, "k_weights") or self.k_weights is None:
            self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1, device=device, dtype=dtype))
        if not hasattr(self, "v_weights") or self.v_weights is None:
            self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1, device=device, dtype=dtype))

        K_new = K_new * self.k_weights.view(1, self.num_kv_heads, 1, 1).to(K_new.dtype)
        V_new = V_new * self.v_weights.view(1, self.num_kv_heads, 1, 1).to(V_new.dtype)

        # -------- RoPE --------
        if not is_cross_attention:
            position_embeddings = kwargs.get("position_embeddings")
            if (position_embeddings is not None
                and isinstance(position_embeddings, (tuple, list))
                and len(position_embeddings) >= 2):
                cos, sin = position_embeddings[0], position_embeddings[1]
                Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin, kwargs.get("position_ids"))
            elif getattr(self, "rotary_emb", None) is not None:
                try:
                    out = self.rotary_emb(hidden_states, position_ids=kwargs.get("position_ids"))
                    if isinstance(out, tuple) and len(out) >= 2:
                        Q, K_new = apply_rotary_pos_emb(Q, K_new, out[0], out[1], kwargs.get("position_ids"))
                except Exception:
                    try:
                        from neuralops.utils.compat import maybe_apply_rotary
                        Q, K_new = maybe_apply_rotary(
                            Q, K_new, getattr(self, "adapter", None),
                            self.rotary_emb, kwargs.get("position_ids"),
                        )
                    except Exception as e:
                        logger.warning("maybe_apply_rotary failed: %s", e)

        # -------- KV-Cache  (FIX 2) --------
        past_k, past_v, past_len = (None, None, 0)
        if not is_cross_attention and past_key_value is not None and past_key_value is not False:
            past_k, past_v, past_len = self._read_past_kv(past_key_value, self.layer_idx)

        if past_k is not None and not is_cross_attention:
            K = torch.cat([past_k.to(K_new.dtype), K_new], dim=-2)
            V = torch.cat([past_v.to(V_new.dtype), V_new], dim=-2)
        else:
            K, V = K_new, V_new

        new_past_key_value: Any = None
        if not is_cross_attention:
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

        scale = d ** -0.5
        is_cross = bool(kwargs.get("is_cross_attention", False)) or is_cross_attention

        # -------- SDPA fast path --------
        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        use_sdpa = (mask is None and head_mask is None and not output_attentions)
        if use_sdpa:
            try:
                _q = Q.half() if Q.dtype == torch.float32 else Q
                _k = K_full.half() if K_full.dtype == torch.float32 else K_full
                _v = V_full.half() if V_full.dtype == torch.float32 else V_full
                _is_causal = self.is_decoder and not is_cross and N_q > 1
                _out = F.scaled_dot_product_attention(
                    _q, _k, _v,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _out = _out.to(dtype).transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
                _out = torch.nan_to_num(_out, nan=0.0, posinf=1e4, neginf=-1e4)
                output = _out if self.is_nested_encoder else self.out_proj(_out)

                valid_m = get_token_valid_mask(attention_mask, B, N_q, device)
                if valid_m is not None:
                    output = output * valid_m.unsqueeze(-1).to(output.dtype)

                if self.is_decoder and not is_cross:
                    return output, new_past_key_value
                return output, None
            except Exception:
                pass

        # -------- Manual path --------
        scores = torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale

        # FIX 3: Bottom-right causal mask when N_q != N_k
        if self.is_decoder and not is_cross:
            if N_q == N_k:
                causal = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool))
            else:
                offset = N_k - N_q
                q_idx = torch.arange(N_q, device=device).unsqueeze(1) + offset
                k_idx = torch.arange(N_k, device=device).unsqueeze(0)
                causal = k_idx <= q_idx
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -10000.0)

        # Padding mask
        eff_mask = encoder_attention_mask if is_cross else mask
        if eff_mask is not None:
            if eff_mask.dim() == 4:
                scores = scores + eff_mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(eff_mask, B, N_k, device)
                if valid_m is not None:
                    scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).to(dtype)
        attn = self.dropout(attn)

        # head_mask
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

        output = context if self.is_nested_encoder else self.out_proj(context)

        valid_m = get_token_valid_mask(attention_mask, B, N_q, device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        attn_ret = attn if output_attentions else None
        if self.is_decoder and not is_cross:
            return output, new_past_key_value
        return output, attn_ret


# =============================================================================
# Injection  (FIX 1: registered in ATTENTION_REGISTRY)
# =============================================================================


@ATTENTION_REGISTRY.register("wgqa")
@ATTENTION_REGISTRY.register("inject_wgqa")
@ATTENTION_REGISTRY.register("wgqa_inject")
def inject_wgqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
    elif (hasattr(target, "bert") and hasattr(target.bert, "encoder")
          and hasattr(target.bert.encoder, "layer")):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_wgqa(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    layer = target

    def _already_injected(m):
        if m is None:
            return False
        if isinstance(m, WeightedGroupedQueryAttention):
            return True
        if "WeightedGroupedQueryAttention" in m.__class__.__name__:
            return True
        inner = getattr(m, "inner", None)
        if isinstance(inner, WeightedGroupedQueryAttention):
            return True
        wrapped = getattr(m, "_wrapped_attention", None)
        if isinstance(wrapped, WeightedGroupedQueryAttention):
            return True
        return False

    if _already_injected(old_attn):
        return None

    # Find src_attn if not provided
    if old_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    old_attn = getattr(cand, "self", cand)
                    break

    if old_attn is None or _already_injected(old_attn):
        return None

    if model_config is None and hasattr(layer, "config"):
        model_config = layer.config
    if model_config is None and old_attn is not None and hasattr(old_attn, "config"):
        model_config = old_attn.config

    hidden_size = (
        _cfg_get(model_config, "hidden_size")
        or _cfg_get(model_config, "d_model")
        or _cfg_get(model_config, "dim")
        or getattr(old_attn, "hidden_size", None)
        or getattr(old_attn, "embed_dim", None)
        or (old_attn.q_proj.in_features if hasattr(old_attn, "q_proj") and hasattr(old_attn.q_proj, "in_features") else None)
        or (old_attn.q_lin.in_features if hasattr(old_attn, "q_lin") and hasattr(old_attn.q_lin, "in_features") else None)
        or (old_attn.query.in_features if hasattr(old_attn, "query") and hasattr(old_attn.query, "in_features") else None)
        or 768
    )
    num_heads = (
        _cfg_get(model_config, "num_attention_heads")
        or _cfg_get(model_config, "num_heads")
        or _cfg_get(model_config, "n_heads")
        or getattr(old_attn, "num_heads", None)
        or getattr(old_attn, "num_attention_heads", None)
        or 12
    )

    if model_config is None:
        import types
        model_config = types.SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(old_attn, "num_key_value_heads", num_heads),
            is_decoder=getattr(old_attn, "is_decoder", False),
        )

    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))
    num_kv_heads = kwargs.pop("num_kv_heads", _cfg_get(comp_config, "num_kv_heads", None))

    try:
        new_attn = WeightedGroupedQueryAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            num_kv_heads=num_kv_heads,
            pretrained_attn=old_attn,
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception as e:
        logger.error("WGQA construction failed: %s", e)
        return None

    # Splice into layer
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
    "WeightedGroupedQueryAttention",
    "inject_wgqa",
    "LoRALinear",
    "apply_rotary_pos_emb",
]