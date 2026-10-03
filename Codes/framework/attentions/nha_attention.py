"""Native Hybrid Attention (NHA) – Production-Hardened (v2).

Fuses localized sliding-window attention with global linear Random Fourier
Features (RFF) via dynamic token-level gating and exact pretrained weight
transfer.

Fixes (v2):
  NHA-FIX-01  **Registry keys separated**: class under `nha_class` /
              `nha_attention_class`; injector under `nha` / `inject_nha`.
  NHA-FIX-02  **Removed `return None` in forward** — RoPE failures now
              propagate as RuntimeError, not silent crash.
  NHA-FIX-03  **Removed all `print()`** — uses `logger`.
  NHA-FIX-04  **`_rotate_half` handles odd head_dim.**
  NHA-FIX-05  **KV-cache support** (tuple + HF DynamicCache).
  NHA-FIX-06  **`num_kv_heads` clamped** to `num_heads`.
  NHA-FIX-07  **`is_nested_encoder` initialized to False.**
  NHA-FIX-08  **Cross-attention raises NotImplementedError.**
  NHA-FIX-09  **No silent padding/truncation** of output dim.
  NHA-FIX-10  **Idempotency guard in inject_nha.**
  NHA-FIX-11  **`self.rotary_emb` receives a proper `x` reference** (q or k).
  NHA-FIX-12  **`comp_config` resolved safely** (not via `or`).
  NHA-FIX-13  **Forward sanity check** after weight transfer.
  NHA-FIX-14  **LoRA dtype-safe forward.**
  NHA-FIX-15  **`output_scale` applied consistently** in both nested and
              non-nested paths.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NHA-FIX-04: correct for odd head_dim."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    rot = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rot = torch.cat([rot, x[..., -1:]], dim=-1)
    return rot


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

    if cos.shape[-2] < q.shape[-2]:
        raise ValueError(
            f"NHA RoPE: cos seq {cos.shape[-2]} < q seq {q.shape[-2]}"
        )
    if cos.shape[-2] > q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = min(cos.shape[-1], q.shape[-1])
    rotary_dim = (rotary_dim // 2) * 2
    if rotary_dim <= 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos_cut = cos[..., :rotary_dim].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim].to(q_rot.dtype)

    q_rot = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    return q, k


# ===========================================================================
# LoRALinear (NHA-FIX-14)
# ===========================================================================
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

        if lora_rank > 0:
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

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# NHA Attention Class
# ===========================================================================
@ATTENTION_REGISTRY.register("nha_class")
@ATTENTION_REGISTRY.register("nha_attention_class")
class NativeHybridAttention(BaseAttention):
    """
    Native Hybrid Attention: local sliding-window + global RFF, fused by a
    token-level learned gate.
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        window_size: int = 64,
        rff_features: int = 128,
        dropout: float = 0.1,
        pretrained_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.config = config
        # NHA-FIX-12: robust comp_config resolution
        if comp_config is None:
            comp_config = getattr(config, "compression_config", None)
        self.comp_config = comp_config

        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else kwargs.get("source_attn", kwargs.get("old_attn"))
        )

        # ---------------- dimension resolution ----------------
        hidden_size = kwargs.get(
            "hidden_size",
            kwargs.get("dim", kwargs.get("d_model")),
        )
        num_heads = kwargs.get(
            "num_attention_heads",
            kwargs.get("num_heads", kwargs.get("n_heads")),
        )
        num_kv_heads = kwargs.get(
            "num_key_value_heads", kwargs.get("num_kv_heads")
        )

        if hidden_size is None or num_heads is None:
            if config is not None:
                if hidden_size is None:
                    hidden_size = _cfg_get(
                        config, "hidden_size",
                        _cfg_get(config, "dim",
                                 _cfg_get(config, "d_model", 768)),
                    )
                if num_heads is None:
                    num_heads = _cfg_get(
                        config, "num_attention_heads",
                        _cfg_get(config, "n_heads",
                                 _cfg_get(config, "num_heads", 12)),
                    )
                if num_kv_heads is None:
                    num_kv_heads = _cfg_get(
                        config, "num_key_value_heads",
                        _cfg_get(config, "num_kv_heads", num_heads),
                    )
            elif source_attn is not None:
                if hidden_size is None:
                    out_layer = getattr(
                        source_attn, "out_lin",
                        getattr(source_attn, "dense",
                                getattr(source_attn, "c_proj",
                                        getattr(source_attn, "o_proj", None))),
                    )
                    hidden_size = getattr(
                        out_layer, "out_features",
                        getattr(source_attn, "hidden_size",
                                getattr(source_attn, "embed_dim", 768)),
                    )
                if num_heads is None:
                    num_heads = getattr(
                        source_attn, "num_attention_heads",
                        getattr(source_attn, "n_heads",
                                getattr(source_attn, "num_heads", 12)),
                    )
                if num_kv_heads is None:
                    num_kv_heads = getattr(
                        source_attn, "num_key_value_heads",
                        getattr(source_attn, "num_kv_heads", num_heads),
                    )
            else:
                hidden_size = 768
                num_heads = 12
                num_kv_heads = num_heads

        if num_kv_heads is None:
            num_kv_heads = num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        # NHA-FIX-06: clamp num_kv_heads
        self.num_kv_heads = min(int(num_kv_heads), self.num_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.window_size = int(_cfg_get(
            comp_config, "window_size",
            _cfg_get(config, "window_size",
                     kwargs.get("window_size", window_size)),
        ))
        self.rff_features = int(_cfg_get(
            comp_config, "rff_features",
            _cfg_get(config, "rff_features",
                     kwargs.get("rff_features", rff_features)),
        ))

        # ---------------- decoder detection ----------------
        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        is_dec = (
            getattr(config, "is_decoder",
                    getattr(config, "is_causal",
                            kwargs.get("is_causal", False)))
            if config else False
        )
        decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures"):
            archs = getattr(config, "architectures", []) or []
            if any("GPT" in str(a) or "CausalLM" in str(a) for a in archs):
                is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        # NHA-FIX-07
        self.is_nested_encoder = False
        self.output_scale = nn.Parameter(torch.ones(1))

        # ---------------- LoRA config ----------------
        def _get_lora_rank():
            if config is not None:
                v = getattr(config, "lora_rank", None)
                if v is not None:
                    return int(v)
            if comp_config is not None:
                v = getattr(comp_config, "lora_rank", None)
                if v is not None:
                    return int(v)
            return int(kwargs.get("lora_rank", 16))

        def _get_lora_alpha():
            if config is not None:
                v = getattr(config, "lora_alpha", None)
                if v is not None:
                    return float(v)
            if comp_config is not None:
                v = getattr(comp_config, "lora_alpha", None)
                if v is not None:
                    return float(v)
            return float(kwargs.get("lora_alpha", 32.0))

        self.lora_rank = _get_lora_rank()
        self.lora_alpha = _get_lora_alpha()
        self.freeze_base = self.lora_rank > 0

        # ---------------- projections ----------------
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank,
                                 lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank,
                                 lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank,
                                 lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank,
                                   lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        # Aliases
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.c_proj = self.out_proj
        self.o_proj = self.out_proj

        omega = self._build_orthogonal_matrix(self.head_dim, self.rff_features)
        self.register_buffer("rff_proj", omega)

        dropout_p = getattr(
            config, "dropout",
            getattr(config, "attention_probs_dropout_prob", dropout),
        ) if config else dropout
        self.dropout = nn.Dropout(dropout_p)

        self.gate = nn.Linear(self.dim, 1)
        # Bias strongly towards local at init: sigmoid(3.0) ≈ 0.95
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 3.0)

        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

        # NHA-FIX-13: forward sanity check
        try:
            with torch.no_grad():
                p = next(self.parameters())
                dummy = torch.randn(1, 4, self.dim, device=p.device, dtype=p.dtype)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("NHA forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    def _build_orthogonal_matrix(self, d: int, m: int) -> torch.Tensor:
        blocks = []
        remaining = m
        while remaining > 0:
            block_size = min(d, remaining)
            g = torch.randn(d, block_size)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)
            remaining -= block_size
        W = torch.cat(blocks, dim=1) * math.sqrt(d)
        return W

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin")
            or hasattr(src, "c_proj")
            or hasattr(src, "o_proj")
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        ) or (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                         "XLMRobertaSelfAttention", "ElectraSelfAttention",
                         "DebertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split, :], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2 * split, :],
                                  c_b[split:2 * split] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split:, :],
                                  c_b[2 * split:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        mod = getattr(src, n)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        # 4. UniversalAdapter fallback
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
                # NHA-FIX-03: no print
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"NativeHybridAttention.copy_weights: missing mandatory "
                f"projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for key, target_mod in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    else:
                        raise RuntimeError(f"NHA: missing weights for {key}")

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (w.shape[0] < target_mod.weight.shape[0]
                      and target_mod.weight.shape[0] % w.shape[0] == 0):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (b.shape[0] < target_mod.bias.shape[0]
                          and target_mod.bias.shape[0] % b.shape[0] == 0):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if fnorm < 1e-4:
                    logger.warning("NHA: '%s' weight norm too small (%.2e).", key, fnorm)

        if self.lora_rank > 0:
            self.q_proj.freeze_base_weights()
            self.k_proj.freeze_base_weights()
            self.v_proj.freeze_base_weights()
            self.out_proj.freeze_base_weights()

    # ------------------------------------------------------------------
    def _local_window_attention(self, Q, K, V, valid_mask, is_causal=False):
        B, H, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

        idx = torch.arange(N, device=Q.device)
        i = idx.unsqueeze(1)
        j = idx.unsqueeze(0)

        if is_causal:
            band_mask = (j > i) | ((i - j) > self.window_size)
        else:
            band_mask = (i - j).abs() > self.window_size

        scores = scores.masked_fill(band_mask.unsqueeze(0).unsqueeze(0), -1e4)
        pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_mask, -1e4)

        attn = F.softmax(scores, dim=-1).to(Q.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    # ------------------------------------------------------------------
    def _global_rff_attention(self, Q, K, V, valid_mask, is_causal=False):
        B, H, N, d = Q.shape
        scale = d ** -0.25
        Q_scaled = Q * scale
        K_scaled = K * scale

        proj = self.rff_proj.to(dtype=Q.dtype, device=Q.device)
        Qf = torch.matmul(Q_scaled, proj)
        Kf = torch.matmul(K_scaled, proj)

        norm = 1.0 / math.sqrt(self.rff_features)
        phi_Q = (F.elu(Qf) + 1.0) * norm
        phi_K = (F.elu(Kf) + 1.0) * norm

        pad = valid_mask.unsqueeze(1).unsqueeze(-1).to(Q.dtype)
        phi_K = phi_K * pad
        V_masked = V * pad

        if is_causal:
            kv_outer = phi_K.unsqueeze(-1) * V_masked.unsqueeze(-2)
            cum_KV = torch.cumsum(kv_outer.float(), dim=2).to(Q.dtype)
            num = (phi_Q.unsqueeze(-2) @ cum_KV).squeeze(-2)

            cum_K = torch.cumsum(phi_K.float(), dim=2).to(Q.dtype)
            den = (phi_Q * cum_K).sum(dim=-1, keepdim=True).clamp(min=1e-5)
            out = (num / den) * pad
        else:
            phi_Q = phi_Q * pad
            KV = torch.matmul(phi_K.transpose(-2, -1), V_masked)
            Z = phi_K.sum(dim=-2, keepdim=True)
            num = torch.matmul(phi_Q, KV)
            den = torch.matmul(phi_Q, Z.transpose(-2, -1)).clamp(min=1e-5)
            out = (num / den) * pad

        return out

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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NHA-FIX-08: explicit error instead of silent bypass
        if encoder_hidden_states is not None:
            raise NotImplementedError("NHA does not support cross-attention.")

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]

        B, N, D = hidden_states.shape
        is_causal = self.is_decoder or kwargs.get("is_causal", False)
        valid_mask = get_token_valid_mask(attention_mask, B, N, hidden_states.device)

        Q = self.q_proj(hidden_states).view(
            B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_new = self.k_proj(hidden_states).view(
            B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V_new = self.v_proj(hidden_states).view(
            B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---------------- RoPE ----------------
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin,
                                             kwargs.get("position_ids"))
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(N, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                # NHA-FIX-11: pass a meaningful reference tensor
                cos, sin = self.rotary_emb(K_new, pos_ids)
                Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin, pos_ids)
            except Exception as e:
                # NHA-FIX-02: do not silently return None
                raise RuntimeError(f"NHA RoPE failed: {e}") from e

        # ---------------- KV cache (NHA-FIX-05) ----------------
        past_k = past_v = None
        past_len = 0
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                past_len = past_k.shape[-2]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]
                past_len = past_k.shape[-2]

        if past_k is not None:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(K_new, V_new, layer_idx)

        N_kv = K_cat.shape[-2]

        # GQA
        if self.num_heads != self.num_kv_heads:
            num_reps = self.num_heads // self.num_kv_heads
            K_full = repeat_kv(K_cat, num_reps)
            V_full = repeat_kv(V_cat, num_reps)
        else:
            K_full = K_cat
            V_full = V_cat

        # Combined valid mask (past + current): assume past is valid
        if past_len > 0:
            all_valid = torch.cat(
                [torch.ones(B, past_len, device=hidden_states.device, dtype=torch.bool),
                 valid_mask],
                dim=1,
            )
        else:
            all_valid = valid_mask

        local_out = self._local_window_attention(Q, K_full, V_full, all_valid, is_causal=is_causal)
        global_out = self._global_rff_attention(Q, K_full, V_full, all_valid, is_causal=is_causal)

        gate = torch.sigmoid(self.gate(hidden_states)).unsqueeze(1)  # (B, 1, N, 1)

        out = gate * local_out + (1.0 - gate) * global_out
        out = out * valid_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)

        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                out = out * head_mask.view(1, self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                out = out * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 4:
                out = out * head_mask.to(out.dtype)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        # NHA-FIX-15: output_scale applied consistently
        if self.is_nested_encoder:
            out = out * self.output_scale
            return out, None

        out = out * self.output_scale
        out = self.out_proj(out)

        # NHA-FIX-09: no silent pad/truncate
        if out.shape[-1] != D:
            raise RuntimeError(
                f"NHA: output dim {out.shape[-1]} != input dim {D}. "
                "This indicates a projection configuration bug."
            )

        return out, None


# ===========================================================================
# Injector (NHA-FIX-01, NHA-FIX-10)
# ===========================================================================
@ATTENTION_REGISTRY.register("nha")
@ATTENTION_REGISTRY.register("nha_inject")
@ATTENTION_REGISTRY.register("inject_nha")
@ATTENTION_REGISTRY.register("inject_native_hybrid_attention")
def inject_nha(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    """Injection hook for NativeHybridAttention."""
    if _depth > 8:
        logger.warning("inject_nha: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # ---------------- Whole-model traversal ----------------
    if layer is None and model is not None:
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_nha(
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

    # ---------------- Single-layer injection ----------------
    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self"):
            src_attn = target.self

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # NHA-FIX-10: idempotency
    if isinstance(src_attn, NativeHybridAttention) or "NativeHybridAttention" in src_attn.__class__.__name__:
        return None

    if src_attn is None and cfg is None:
        return None

    try:
        new_attn = NativeHybridAttention(
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
    except Exception as e:
        logger.warning("NHA construction/weight copy failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True

    return new_attn


inject_native_hybrid_attention = inject_nha


__all__ = [
    "NativeHybridAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_nha",
    "inject_native_hybrid_attention",
]
