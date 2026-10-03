"""
GoldFinch-AHN: Adaptive Hybrid Network — Sliding Window + Gated Recurrent Memory.
Production-Hardened (v2).

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

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
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
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ===========================================================================
# LoRALinear (dtype-safe)
# ===========================================================================
class LoRALinear(nn.Module):
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
# GF-FIX-06: proper gated recurrent memory cell
# ===========================================================================
class GatedRecurrentMemory(nn.Module):
    """
    A light gated recurrent memory:
        g_t = sigmoid(W_g x_t + b_g)
        h_t = (1 - g_t) * h_{t-1} + g_t * tanh(W_x x_t + b_x)
        y_t = h_t

    Not a full DeltaNet but has the required gate structure and behaves as
    a proper recurrent memory (unlike the previous plain Linear layer).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.value_proj = nn.Linear(d_model, d_model, bias=True)
        # Start gate slightly open so that memory doesn't dominate attention
        nn.init.constant_(self.gate_proj.bias, -1.0)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        target_dtype = x.dtype
        gw = self.gate_proj.weight.to(target_dtype)
        gb = self.gate_proj.bias.to(target_dtype) if self.gate_proj.bias is not None else None
        vw = self.value_proj.weight.to(target_dtype)
        vb = self.value_proj.bias.to(target_dtype) if self.value_proj.bias is not None else None
        g = torch.sigmoid(F.linear(x, gw, gb))
        v = torch.tanh(F.linear(x, vw, vb))
        # Parallel scan over sequence (small L → simple python loop)
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=target_dtype)
        outs = []
        for t in range(L):
            h = (1.0 - g[:, t]) * h + g[:, t] * v[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1).to(target_dtype)


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "goldfinch_ahn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "goldfinch", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class GoldFinchAHNBlock(BaseAttention):
    """
    GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.

    Semantics of `window_size`: full window width (in tokens). Effective
    radius = `window_size // 2`, so `window_size = 64` → ±32 tokens.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        window_size: int = 64,
        is_causal: bool = False,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        gate_init_bias: float = 1.8,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # GF-FIX-05: None-safe
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- hidden size ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None:
                if config is not None:
                    d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
                elif source_attn is not None:
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
                        for attr in ("q_proj", "query", "q_lin"):
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

        # ---- num_heads ----
        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None:
                if config is not None:
                    num_heads = getattr(
                        config, "num_attention_heads", getattr(config, "n_heads", 12)
                    )
                elif source_attn is not None:
                    num_heads = getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "n_heads", 12),
                    )
                else:
                    num_heads = 12

        self.dim = int(d_model)
        self.hidden_size = self.dim
        self.num_heads = max(1, int(num_heads))

        # ---- num_kv_heads ----
        if config is not None:
            self.num_kv_heads = getattr(
                config,
                "num_key_value_heads",
                getattr(config, "num_kv_heads", kwargs.get("num_kv_heads", self.num_heads)),
            )
        elif source_attn is not None:
            self.num_kv_heads = (
                getattr(source_attn, "num_key_value_heads", None)
                or getattr(source_attn, "num_kv_heads", None)
                or kwargs.get("num_kv_heads", self.num_heads)
            )
        else:
            self.num_kv_heads = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(self.num_kv_heads), self.num_heads))

        # ---- head_dim ----
        _head_dim = getattr(config, "head_dim", None) if config else None
        if _head_dim is None and source_attn is not None:
            _head_dim = getattr(source_attn, "head_dim", None)
        if _head_dim is None:
            _head_dim = self.dim // self.num_heads
        self.head_dim = int(_head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ---- window (GF-FIX-09) ----
        if config is not None:
            self.window_size = int(
                getattr(
                    config,
                    "window_size",
                    (
                        getattr(
                            comp_config,
                            "window_size",
                            kwargs.get("window_size", window_size),
                        )
                        if comp_config is not None
                        else window_size
                    ),
                )
            )
        elif comp_config is not None:
            self.window_size = int(
                getattr(comp_config, "window_size", kwargs.get("window_size", window_size))
            )
        else:
            self.window_size = int(kwargs.get("window_size", window_size))
        self.window_radius = max(1, self.window_size // 2)

        # ---- decoder detection ----
        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False) or getattr(config, "is_causal", False)
            )
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
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False) or is_causal:
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(source_attn, "is_decoder", False):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- LoRA config ----
        if config is not None:
            self.lora_rank = int(
                getattr(
                    config,
                    "lora_rank",
                    (
                        getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
                        if comp_config is not None
                        else kwargs.get("lora_rank", 16)
                    ),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config,
                    "lora_alpha",
                    (
                        getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
                        if comp_config is not None
                        else kwargs.get("lora_alpha", 32.0)
                    ),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
            )
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))

        self.freeze_base = self.lora_rank > 0
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )

        # ---- projections ----
        self.W_q = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_v = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # GF-FIX-06: proper gated recurrent memory
        self.delta_mem = GatedRecurrentMemory(self.dim)

        # per-head attention / memory gate
        self.gate = nn.Linear(self.dim, self.num_heads)
        with torch.no_grad():
            self.gate.bias.data.fill_(gate_init_bias)
            nn.init.zeros_(self.gate.weight)

        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = 0.1
        if config is not None:
            dropout_p = float(
                getattr(
                    config,
                    "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                )
            )
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        # GF-FIX-11: single output scale, initialized to preserve pretrained signal
        self.output_scale = nn.Parameter(torch.ones(1))

        # Storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.W_q

    @property
    def key(self):
        return self.W_k

    @property
    def value(self):
        return self.W_v

    @property
    def q_proj(self):
        return self.W_q

    @property
    def k_proj(self):
        return self.W_k

    @property
    def v_proj(self):
        return self.W_v

    @property
    def q_lin(self):
        return self.W_q

    @property
    def k_lin(self):
        return self.W_k

    @property
    def v_lin(self):
        return self.W_v

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

    def freeze_base_weights(self):
        for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
            if hasattr(m, "freeze_base_weights"):
                m.freeze_base_weights()

    def unfreeze_base_weights(self):
        for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
            if hasattr(m, "unfreeze_base_weights"):
                m.unfreeze_base_weights()

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        is_nested = False
        if (
            cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            )
            or hasattr(src, "query")
            and not has_out
            or hasattr(src, "self")
            and hasattr(src, "output")
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    "Conv1D" in c_attn_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
                extracted["k"] = (
                    c_w[split : 2 * split],
                    c_b[split : 2 * split] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split :],
                    c_b[2 * split :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        "Conv1D" in c_proj_mod.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # Nested (BERT / RoBERTa)
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

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
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data.clone(),
                                b.data.clone() if b is not None else None,
                            )
                            break

        # UniversalAdapter fallback
        missing = [
            k
            for k, (w, _) in extracted.items()
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
                logger.debug("Goldfinch UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                "Goldfinch copy_weights: missing %s; using initialized weights",
                missing_after,
            )

        target_map = {"q": self.W_q, "k": self.W_k, "v": self.W_v, "o": self.out_proj}

        with torch.no_grad():
            for key, target in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(b[:mb].to(target.bias.device, target.bias.dtype))
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                # GF-FIX-02: warn, do not raise
                w_norm = float(torch.linalg.norm(target.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "Goldfinch copy_weights: '%s' has near-zero norm (%.2e)",
                        key,
                        w_norm,
                    )

                if self.freeze_base:
                    target.freeze_base_weights()

        # Track compressed storage
        try:
            for m in (self.W_q, self.W_k, self.W_v, self.out_proj, self.delta_mem):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # GF-FIX-15: freeze once at the end, only if needed
        if self.freeze_base:
            self.freeze_base_weights()

    @classmethod
    def from_distilbert_pretrained(cls, old_attn, model_config=None, **kwargs):
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(attention_mask, B, L, device):
        if attention_mask is None:
            return None

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
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2).squeeze(1)
            if m.shape == (B, L):
                return m

        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            pass
        return None

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ):
        # GF-FIX-04: check parameter directly
        if encoder_hidden_states is not None:
            raise NotImplementedError("GoldFinchAHN does not support cross-attention.")

        x = hidden_states
        if not x.is_contiguous():
            x = x.contiguous()
        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        device = x.device
        dtype_in = x.dtype

        # ---- QKV ----
        q = self.W_q(x).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.W_k(x).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # ---- RoPE ----
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(k, seq_len=L)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(k, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Sliding-window attention ----
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * (d**-0.5)

        idx = torch.arange(L, device=device)
        diff = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        band = diff <= self.window_radius

        if self.is_causal:
            causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
            valid_local = causal & band
        else:
            valid_local = band

        scores = scores.masked_fill(~valid_local.unsqueeze(0).unsqueeze(0), -1e4)

        valid_mask = self._normalize_valid_mask(attention_mask, B, L, device)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores, dim=-1).to(dtype_in)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        # GF-FIX-03 / GF-FIX-13: head_mask applied to per-head attention
        if head_mask is not None:
            hm = head_mask.to(attn_weights.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            attn_weights = attn_weights * hm

        short_out = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, self.all_head_size)
        )

        # ---- Recurrent memory ----
        long_out = self.delta_mem(x).to(dtype_in)  # (B, L, D)

        # ---- Gated fusion ----
        # gate: (B, L, H) → (B, L, 1) after mean
        gate_w = self.gate.weight.to(dtype_in)
        gate_b = self.gate.bias.to(dtype_in) if self.gate.bias is not None else None
        gate = (
            torch.sigmoid(F.linear(x, gate_w, gate_b)).mean(dim=-1, keepdim=True).to(dtype_in)
        )  # (B, L, 1)

        # Pad long_out to all_head_size if needed (should match since dim = all_head_size for MHA)
        if long_out.shape[-1] != self.all_head_size:
            if long_out.shape[-1] < self.all_head_size:
                long_out = F.pad(long_out, (0, self.all_head_size - long_out.shape[-1]))
            else:
                long_out = long_out[..., : self.all_head_size]

        fused = (gate * short_out.to(dtype_in) + (1.0 - gate) * long_out).to(dtype_in)

        # Zero-out padded positions
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1).to(fused.dtype)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
            # GF-FIX-11: apply output_scale once
            out = (out * self.output_scale.to(out.dtype)).to(out.dtype)

        return out.to(dtype_in), (attn_weights if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("W_q", self.W_q),
            ("W_k", self.W_k),
            ("W_v", self.W_v),
            ("out_proj", self.out_proj),
            ("delta_mem", self.delta_mem),
            ("gate", self.gate),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["output_scale"] = _sz(self.output_scale)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("goldfinch_ahn_inject")
@ATTENTION_REGISTRY.register("inject_goldfinch_ahn")
def inject_goldfinch_ahn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> GoldFinchAHNBlock | nn.Module | None:
    if _depth > 8:
        logger.warning("inject_goldfinch_ahn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_goldfinch_ahn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---- Single-layer ----
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # GF-FIX-08: idempotency guard
    if src_attn is not None and (
        isinstance(src_attn, GoldFinchAHNBlock) or "GoldFinch" in src_attn.__class__.__name__
    ):
        return None
    if src_attn is not None and (
        "DeltaNet" in src_attn.__class__.__name__
        or "LinearAttention" in src_attn.__class__.__name__
    ):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn.q_proj, "in_features", None) if hasattr(src_attn, "q_proj") else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        kwargs.pop("d_model", None)
        kwargs.pop("num_heads", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("num_attention_heads", None)
        kwargs.pop("old_attn", None)
        new_attn = GoldFinchAHNBlock(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)
    except Exception as e:
        logger.error("GoldFinchAHN injection failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn

    return new_attn


__all__ = [
    "GoldFinchAHNBlock",
    "inject_goldfinch_ahn",
    "GatedRecurrentMemory",
    "LoRALinear",
    "apply_rotary_pos_emb",
]
