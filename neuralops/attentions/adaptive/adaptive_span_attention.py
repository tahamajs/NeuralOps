"""Adaptive Span Attention

"""

from __future__ import annotations
from neuralops.core.attention_utils import normalize_attention_mask as _normalize_mask

from neuralops.core.attention_utils import (
    safe_mask_value as _safe_mask_value,
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSpanConfig:
    """Runtime configuration for AdaptiveSpanAttention."""
    max_span: int = 128
    min_span: float = 32.0
    ramp_size: float = 32.0
    dropout: float = 0.1
    lora_rank: int = 16
    lora_alpha: float = 32.0
    rolling_cache_size: int | None = None


def get_total_span_loss(model: nn.Module, lambda_span: float = 0.01) -> torch.Tensor:
    """L1 span regularization across all AdaptiveSpanAttention modules.

    L_span = lambda * (1/|M|) * sum_m mean_h(sigmoid(logit_mh))
    """
    span_modules = getattr(model, "_adaptive_span_modules", None)
    if span_modules is None:
        span_modules = [m for m in model.modules() if isinstance(m, AdaptiveSpanAttention)]

    losses: list[torch.Tensor] = []
    for m in span_modules:
        if hasattr(m, "get_span_regularization_loss"):
            losses.append(m.get_span_regularization_loss(lambda_span=lambda_span))

    if not losses:
        try:
            return next(model.parameters()).new_zeros(())
        except StopIteration:
            return torch.zeros((), dtype=torch.float32)

    return torch.stack(losses).mean()




def _rotate_half_q(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dim by half. Cached canonical version."""
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def _gather_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract cos/sin values at given position_ids.

    [FIX-1] Handles 2D, 3D, AND 4D cos/sin.
    """
    # Normalize to (Bc, Hc, S, d)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 4:
        # Already (B, H, S, d) — no-op
        pass
    else:
        raise ValueError(f"Unsupported cos.dim()={cos.dim()}")

    Bc, Hc, S, d = cos.shape
    ids = position_ids.clamp(min=0, max=S - 1)  # (B, N)

    if Bc == 1:
        # Broadcast cos/sin across batch
        cos_g = cos[0, 0][ids]  # (B, N, d)
        sin_g = sin[0, 0][ids]
        cos_g = cos_g.unsqueeze(1)  # (B, 1, N, d)
        sin_g = sin_g.unsqueeze(1)
    else:
        cos_flat = cos[:, 0].contiguous()  # (B, S, d)
        sin_flat = sin[:, 0].contiguous()
        cos_g = torch.gather(
            cos_flat, 1, ids.unsqueeze(-1).expand(-1, -1, d)
        ).unsqueeze(1)  # (B, 1, N, d)
        sin_g = torch.gather(
            sin_flat, 1, ids.unsqueeze(-1).expand(-1, -1, d)
        ).unsqueeze(1)
    return cos_g, sin_g


def _build_keep_mask(
    B: int,
    N_q: int,
    N_k: int,
    is_causal: bool,
    attention_mask: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Combine causal + padding + encoder masks into a single bool (B, 1, N_q, N_k)."""
    mask: torch.Tensor | None = None

    if is_causal:
        if N_q == N_k:
            causal = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool))
        else:
            offset = N_k - N_q
            q_idx = torch.arange(N_q, device=device).unsqueeze(1) + offset
            k_idx = torch.arange(N_k, device=device).unsqueeze(0)
            causal = k_idx <= q_idx
        mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, N_q, N_k)

    def _combine(am: torch.Tensor) -> torch.Tensor:
        nonlocal mask
        if am.dim() == 2:
            am_b = am[:, None, None, :]
        elif am.dim() == 3:
            am_b = am.unsqueeze(1) if am.shape[1] != N_q else am[:, None, :, :]
        elif am.dim() == 4:
            am_b = am
        else:
            return (
                mask
                if mask is not None
                else torch.ones((B, 1, N_q, N_k), device=device, dtype=torch.bool)
            )

        if am_b.dtype != torch.bool:
            if am_b.max() <= 1.0 and am_b.min() >= 0.0:
                am_b = am_b.bool()
            else:
                am_b = am_b > -1e3
        return am_b

    if attention_mask is not None:
        am = _combine(attention_mask)
        mask = am if mask is None else (mask & am)

    if encoder_attention_mask is not None:
        em = _combine(encoder_attention_mask)
        mask = em if mask is None else (mask & em)

    return mask






class IdentityOutProj(nn.Identity):
    """Placeholder for nested encoders where out_proj is external."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_A = None
        self.lora_B = None

    def freeze_base_weights(self) -> None:
        pass


# =============================================================================
# Adaptive Span Attention
# =============================================================================


@ATTENTION_REGISTRY.register("adaptive_span_attention")
class AdaptiveSpanAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dropout: float = 0.1,
        max_span: int = 128,
        ramp_size: float = 32.0,
        lora_rank: int = 16,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        layer_idx: int | None = None,
        return_style: str | None = None,
        rolling_cache_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # ---------------- Dimensions ----------------
        hidden_size = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
        num_heads = (
            kwargs.get("num_heads") or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        )
        num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")
        head_dim = 0

        if config is not None:
            adapter = ModelAdapter(config)
            hidden_size = hidden_size or adapter.hidden_size
            num_heads = num_heads or adapter.num_heads
            num_kv_heads = num_kv_heads or adapter.num_key_value_heads
            head_dim = getattr(adapter, "head_dim", 0)

        if not hidden_size and source_attn is not None:
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
                cand = getattr(source_attn.q_proj, "in_features", None)
            hidden_size = cand

        hidden_size = int(hidden_size or 768)
        num_heads = int(num_heads or 12)
        num_kv_heads = int(num_kv_heads or num_heads)
        head_dim = head_dim if head_dim > 0 else (hidden_size // num_heads if num_heads > 0 else 64)

        # [FIX-9] Single source of truth: `hidden_size`. `d_model` is an alias property.
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.layer_idx: int = int(layer_idx) if layer_idx is not None else 0
        self.rolling_cache_size: int | None = rolling_cache_size

        # ---------------- Span config ----------------
        cfg_max_span = getattr(config, "max_span", max_span) if config else max_span
        cfg_ramp = getattr(config, "ramp_size", ramp_size) if config else ramp_size
        if comp_config is not None:
            cfg_max_span = getattr(comp_config, "max_span", cfg_max_span)
            cfg_ramp = getattr(comp_config, "ramp_size", cfg_ramp)
        self.max_span: int = int(cfg_max_span)
        self.ramp_size: float = float(cfg_ramp)
        self.min_span: float = float(
            kwargs.get("min_span", getattr(config, "min_span", 32.0) if config else 32.0)
        )

        # ---------------- Decoder / nested encoder detection ----------------
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = (
            getattr(
                config,
                "is_decoder",
                getattr(config, "is_causal", kwargs.get("is_causal", False)),
            )
            if config
            else False
        )
        _dec_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
            "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if any(model_type.startswith(p) for p in _dec_prefixes):
            is_dec = True
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a)
            for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        source_cls = source_attn.__class__.__name__ if source_attn is not None else ""
        cfg_model_type = getattr(config, "model_type", "").lower() if config else ""
        is_nested = (
            not is_dec
            and (
                source_cls in ("BertSelfAttention", "RobertaSelfAttention")
                or cfg_model_type in ("roberta", "bert")
                or (
                    source_cls.endswith(("SelfAttention", "SelfAttn"))
                    and "distil" not in source_cls.lower()
                    and "gpt" not in source_cls.lower()
                    and "qwen" not in source_cls.lower()
                )
            )
        )
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", is_nested))

        if return_style is None:
            if self.is_nested_encoder:
                return_style = "bert"
            elif "GPT2" in source_cls or "gpt2" in cfg_model_type:
                return_style = "gpt2"
            elif "T5" in source_cls or "t5" in cfg_model_type:
                return_style = "t5"
            elif self.is_decoder:
                return_style = "llama"
            else:
                return_style = "bert"
        self.return_style: str = str(return_style)

        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: nn.Embedding | None = None
        self.relative_attention_num_buckets: int = (
            getattr(config, "relative_attention_num_buckets", 32) if config else 32
        )
        self.relative_attention_max_distance: int = (
            getattr(config, "relative_attention_max_distance", 128) if config else 128
        )

        # ---------------- LoRA config ----------------
        if config is not None:
            resolved_lora_rank = getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
            resolved_lora_alpha = getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        elif comp_config is not None:
            resolved_lora_rank = getattr(
                comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)
            )
            resolved_lora_alpha = getattr(
                comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)
            )
        else:
            resolved_lora_rank = kwargs.get("lora_rank", lora_rank)
            resolved_lora_alpha = kwargs.get("lora_alpha", 32.0)

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            for p in source_attn.parameters():
                target_dtype = p.dtype
                break

        self.lora_rank = int(resolved_lora_rank)
        self.lora_alpha = float(resolved_lora_alpha)
        self.freeze_base: bool = self.lora_rank > 0

        # ---------------- Projections ----------------
        self.q_proj = LoRALinear(
            self.hidden_size, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.hidden_size, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base, dtype=target_dtype,
        )
        if self.is_nested_encoder:
            self.out_proj: nn.Module = IdentityOutProj()
        else:
            self.out_proj = LoRALinear(
                self.all_head_size, self.hidden_size, bias=True,
                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                freeze_base=self.freeze_base, dtype=target_dtype,
            )

        # ---------------- Span logits ----------------
        # Init at -1.5 (span ≈ 0.18 * max_span ≈ 23 tokens) so compression takes active effect
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), -1.5, dtype=torch.float32))

        # ---------------- Dropout ----------------
        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", dropout),
                ),
            )
            if config
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )

        if source_attn is not None:
            self.copy_weights(source_attn)

        # Convert only if target_dtype was explicitly specified AND weights were not copied
        if target_dtype is not None and source_attn is None:
            self.q_proj.to(target_dtype)
            self.k_proj.to(target_dtype)
            self.v_proj.to(target_dtype)
            self.out_proj.to(target_dtype)

    # ------------------------------------------------------------------
    # Property aliases (backward compat + HF conventions)
    # ------------------------------------------------------------------
    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def current_span(self) -> torch.Tensor:
        raw_span = torch.sigmoid(self.span_logits) * self.max_span
        return torch.clamp(raw_span, min=self.min_span)

    def get_span_loss(self, lambda_span: float = 0.01) -> torch.Tensor:
        return self.get_span_regularization_loss(lambda_span)

    def get_span_regularization_loss(self, lambda_span: float = 0.01) -> torch.Tensor:
        return lambda_span * torch.sigmoid(self.span_logits).mean()

    # ------------------------------------------------------------------
    # Relative position bias (T5-style)
    # ------------------------------------------------------------------
    def _relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        rel_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        rel_large = torch.min(rel_large, torch.full_like(rel_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, rel_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device=None) -> torch.Tensor:
        if self.relative_attention_bias is None:
            return torch.zeros((1, self.num_heads, query_length, key_length), device=device)
        ctx = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        mem = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        rel = mem - ctx
        buckets = self._relative_position_bucket(
            rel,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute([2, 0, 1]).unsqueeze(0)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. Try UniversalAdapter (handles most HF models)
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback for %s: %s", proj, e)

        # 2. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c_attn_layer = src.c_attn
            base_c = getattr(c_attn_layer, "base_layer", c_attn_layer)
            weight = getattr(base_c, "weight", getattr(c_attn_layer, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_layer, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    hasattr(c_attn_layer, "nf")
                    or hasattr(base_c, "nf")
                    or c_attn_layer.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                )
                if is_conv1d and weight.dim() == 2:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split].contiguous(),
                    bias_data[:split].contiguous() if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split].contiguous(),
                    bias_data[split : 2 * split].contiguous() if bias_data is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :].contiguous(),
                    bias_data[2 * split :].contiguous() if bias_data is not None else None,
                )

        # 3. GPT-2 c_proj
        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_layer = src.c_proj
            base_o = getattr(c_proj_layer, "base_layer", c_proj_layer)
            w_o = getattr(base_o, "weight", getattr(c_proj_layer, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    hasattr(c_proj_layer, "nf")
                    or hasattr(base_o, "nf")
                    or c_proj_layer.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                )
                if is_conv1d and w_o.dim() == 2:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_layer, "bias", None))
                extracted["o"] = (
                    w_o.contiguous(),
                    b_o.data.clone().contiguous() if b_o is not None else None,
                )

        # 4. BERT / RoBERTa nested
        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        # 5. BERT output.dense
        if (
            hasattr(src, "output")
            and getattr(src, "output", None) is not None
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 6. Named fallback
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Determine nested encoder
        cls_name = src.__class__.__name__
        cfg_type = getattr(self.config, "model_type", "").lower() if self.config else ""
        if not self.is_nested_encoder:
            self.is_nested_encoder = bool(
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or cfg_type in ("roberta", "bert")
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                    and "gpt" not in cls_name.lower()
                    and "qwen" not in cls_name.lower()
                )
            )
        if self.is_nested_encoder and not isinstance(self.out_proj, IdentityOutProj):
            self.out_proj = IdentityOutProj()

        # Verify Frobenius norm of extracted weights or source module
        if hasattr(src, "weight") and torch.norm(src.weight.float()).item() == 0.0:
            raise RuntimeError("Source weight tensor has zero Frobenius norm")
        for k in ("q", "k", "v", "o"):
            w, _ = extracted[k]
            if w is not None and torch.norm(w.float()).item() == 0.0:
                raise RuntimeError(f"Weight tensor for {k} has zero Frobenius norm")

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("AdaptiveSpanAttention missing weights for: %s", missing)

        # Copy into targets
        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                if proj_key == "o" and self.is_nested_encoder:
                    continue
                w, b = extracted[proj_key]
                if w is not None:
                    out_d, in_d = target.weight.shape
                    src_out, src_in = w.shape
                    c_o, c_i = min(out_d, src_out), min(in_d, src_in)
                    target.weight.data[:c_o, :c_i].copy_(w[:c_o, :c_i])
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        c_b = min(len(target.bias), len(b_val))
                        target.bias.data[:c_b].copy_(b_val[:c_b])
                if hasattr(target, "freeze_base_weights"):
                    target.freeze_base_weights()

        # Extra config copy
        if getattr(src, "rotary_emb", None) is not None:
            self.rotary_emb = src.rotary_emb
        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if getattr(src, "relative_attention_bias", None) is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_len(past_key_value: Any, layer_idx: int) -> int:
        if past_key_value is None or past_key_value is False:
            return 0
        if hasattr(past_key_value, "get_seq_length"):
            try:
                return int(past_key_value.get_seq_length(layer_idx))
            except Exception:
                try:
                    return int(past_key_value.get_seq_length())
                except Exception:
                    pass
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k = past_key_value[0]
            if k is not None:
                return int(k.shape[-2])
        return 0

    def _update_cache(
        self, K_new: torch.Tensor, V_new: torch.Tensor, past_key_value: Any
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Returns (K, V, new_cache)."""
        if past_key_value is None or past_key_value is False:
            return K_new, V_new, None

        # HF DynamicCache
        if hasattr(past_key_value, "update"):
            try:
                result = past_key_value.update(K_new, V_new, self.layer_idx)
                if isinstance(result, tuple) and len(result) == 3:
                    return result[0], result[1], result[2]
                # Some versions return just the cache
                K = past_key_value.key_cache[self.layer_idx]
                V = past_key_value.value_cache[self.layer_idx]
                return K, V, past_key_value
            except Exception as e:
                logger.debug("DynamicCache.update failed: %s", e)

        # Tuple (K, V)
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            past_k, past_v = past_key_value[0], past_key_value[1]
            if past_k is not None:
                K = torch.cat([past_k, K_new], dim=-2)
                V = torch.cat([past_v, V_new], dim=-2)
            else:
                K, V = K_new, V_new
            return K, V, (K, V)

        return K_new, V_new, (K_new, V_new)

    # ------------------------------------------------------------------
    # Span penalty (FP32, batch-aware, sync-free)
    # ------------------------------------------------------------------
    def _compute_span_penalty(
        self,
        scores: torch.Tensor,
        position_ids: torch.Tensor | None,
        N_k: int,
        device: torch.device,
        is_causal: bool,
    ) -> torch.Tensor:
        """[FIX-3, FIX-5, FIX-6, FIX-8] Compute span penalty in FP32.

        Returns the penalty tensor to ADD to scores (in scores.dtype).
        """
        H = self.num_heads
        B, H, N_q, N_k = scores.shape

        effective_min = min(self.min_span, float(N_k))
        raw_span = torch.sigmoid(self.span_logits) * self.max_span  # (H,)
        span = raw_span.clamp(min=effective_min)  # (H,)

        # [FIX-8, FIX-6] Use full position_ids (batch-aware), no .item()
        if position_ids is not None:
            q_pos = position_ids.to(torch.float32)  # (B, N_q)
            if N_k == N_q:
                k_pos = q_pos  # (B, N_k)
            else:
                # Decode: last q_pos is anchor, walk backwards
                # k_pos[b, j] = q_pos[b, N_q-1] - (N_k - 1 - j)
                anchor = q_pos[:, -1:]  # (B, 1)
                offsets = torch.arange(N_k - 1, -1, -1, device=device, dtype=torch.float32)  # (N_k,)
                k_pos = anchor - offsets.unsqueeze(0)  # (B, N_k)
        else:
            q_pos = torch.arange(N_q, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
            k_pos = torch.arange(N_k, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)

        # Try fused Triton kernel in SRAM
        try:
            from neuralops.core.triton_kernels.elementwise import fused_adaptive_span_penalty_triton
            penalty = fused_adaptive_span_penalty_triton(
                span, q_pos, k_pos, float(self.ramp_size), is_causal=is_causal
            )
            return penalty.to(scores.dtype)
        except Exception:
            pass

        # dist: (B, N_q, N_k)
        if is_causal:
            dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).clamp(min=0.0)
        else:
            dist = (q_pos.unsqueeze(2) - k_pos.unsqueeze(1)).abs()

        # span_h: (H, 1, 1) → ramp: (H, B, N_q, N_k) → log: same
        span_h = span.view(H, 1, 1).unsqueeze(1)  # (H, 1, 1, 1)
        dist_e = dist.unsqueeze(0)  # (1, B, N_q, N_k)
        ramp = ((self.ramp_size + span_h - dist_e) / self.ramp_size).clamp(0.0, 1.0)
        span_mask = torch.log(ramp.clamp(min=1e-9))  # (H, B, N_q, N_k) FP32

        # Reorder to (B, H, N_q, N_k) then cast to scores.dtype
        penalty = span_mask.permute(1, 0, 2, 3).contiguous()
        return penalty.to(scores.dtype)

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
    ):
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, _ = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ---- Projections ----
        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K_new = (
            self.k_proj(kv_input).view(B, -1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        )
        V_new = (
            self.v_proj(kv_input).view(B, -1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        )

        # ---- Position IDs ----
        position_ids = kwargs.get("position_ids")
        past_len = self._read_past_len(past_key_value, self.layer_idx) if not is_cross_attention else 0
        if position_ids is None and not is_cross_attention:
            position_ids = (
                torch.arange(past_len, past_len + N_q, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )

        # ---- RoPE ----
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin, position_ids=position_ids)
        elif getattr(self, "rotary_emb", None) is not None:
            if position_ids is None:
                position_ids = (
                    torch.arange(N_q, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
                )
            try:
                cos, sin = self.rotary_emb(K_new, position_ids=position_ids)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(K_new, position_ids)
                except Exception:
                    cos, sin = None, None
            except Exception:
                cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin, position_ids=position_ids)

        # ---- KV cache ----
        use_cache = kwargs.get("use_cache", False) or past_key_value is not None
        if is_cross_attention:
            K, V = K_new, V_new
            new_past_key_value = past_key_value
        else:
            K, V, new_past_key_value = self._update_cache(K_new, V_new, past_key_value)
            # Rolling cache
            if (
                self.rolling_cache_size is not None
                and K.shape[-2] > self.rolling_cache_size
            ):
                K = K[..., -self.rolling_cache_size :, :].contiguous()
                V = V[..., -self.rolling_cache_size :, :].contiguous()
                if isinstance(new_past_key_value, tuple):
                    new_past_key_value = (K, V)

        N_k = K.shape[-2]

        # ---- GQA ----
        if self.kv_repeat > 1:
            if K.shape[1] == self.num_kv_heads:
                K_full = repeat_kv(K, self.kv_repeat)
                V_full = repeat_kv(V, self.kv_repeat)
            elif K.shape[1] == self.num_heads:
                K_full, V_full = K, V
            else:
                raise ValueError(f"Unexpected K head dim {K.shape[1]}")
        else:
            K_full, V_full = K, V

        # ---- Scores (in FP32 for stability) ----
        scale = d**-0.5
        scores = (torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale)

        # ---- Relative position bias ----
        position_bias = kwargs.get("position_bias")
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=device)
        if position_bias is not None:
            scores = scores + position_bias.float()

        # ---- [FIX-3] Span penalty in FP32 ----
        if not is_cross_attention:
            is_causal_here = self.is_decoder or kwargs.get("is_causal", False)
            penalty = self._compute_span_penalty(
                scores, position_ids, N_k, device, is_causal_here
            )
            scores = scores + penalty.float()

        # ---- [FIX-4] Combined mask ----
        is_causal = self.is_decoder and not is_cross_attention
        keep_mask = _build_keep_mask(
            B=B, N_q=N_q, N_k=N_k, is_causal=is_causal,
            attention_mask=attention_mask,
            encoder_attention_mask=(encoder_attention_mask if is_cross_attention else None),
            device=device,
        )

        fully_masked: torch.Tensor | None = None
        if keep_mask is not None:
            fully_masked = ~keep_mask.any(dim=-1, keepdim=True)  # (B, 1, N_q, 1)
            mask_val = _safe_mask_value(scores.dtype)
            scores = scores.masked_fill(~keep_mask, mask_val)

        if fully_masked is not None:
            scores = torch.where(fully_masked, torch.zeros_like(scores), scores)

        # ---- Softmax (in FP32) ----
        attn = F.softmax(scores.float(), dim=-1)

        if fully_masked is not None:
            attn = attn.masked_fill(fully_masked, 0.0)

        attn = attn.to(dtype)
        attn = self.dropout(attn)

        # ---- [FIX-2] head_mask: proper broadcasting ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, H, 1, 1)  # (1, H, 1, 1)
            elif head_mask.dim() == 2:
                # Could be (B, H) or (1, H) — normalize to (B, H, 1, 1)
                head_mask = head_mask.view(head_mask.shape[0], H, 1, 1)
            elif head_mask.dim() == 3:
                # (B, N_q, H) → (B, H, N_q, 1)
                head_mask = head_mask.transpose(1, 2).unsqueeze(-1)
            # head_mask.dim() == 4 → already (B, H, N_q, N_k) or (B, H, 1, 1)
            attn = attn * head_mask.to(attn.dtype)

        # ---- Context ----
        context = torch.matmul(attn, V_full.to(attn.dtype)).to(dtype)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        # ---- Format output ----
        return self._format_output(
            output,
            attn if output_attentions else None,
            new_past_key_value,
            position_bias,
            output_attentions,
        )

    def _format_output(
        self,
        output: torch.Tensor,
        attn: torch.Tensor | None,
        past_kv: Any | None,
        position_bias: torch.Tensor | None,
        output_attentions: bool,
    ):
        style = self.return_style
        if style == "bert":
            return output, attn
        if style == "gpt2":
            if output_attentions:
                return output, past_kv, attn
            return output, past_kv
        if style == "t5":
            if output_attentions:
                return output, position_bias, attn
            return output, position_bias
        if output_attentions:
            return output, attn, past_kv
        return output, past_kv


# =============================================================================
# Injector
# =============================================================================


def inject_adaptive_span(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> AdaptiveSpanAttention | nn.Module | None:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Detect layer list ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("layers",),
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
        injected: list[nn.Module] = []
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_span(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                injected.append(res)
        if injected:
            if hasattr(target, "_adaptive_span_modules"):
                target._adaptive_span_modules.extend(injected)
            else:
                target._adaptive_span_modules = injected
            return target
        return None

    # ---- Find source attention ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                src_attn = getattr(cand, "self", cand)
                break
    if src_attn is None and hasattr(target, "forward"):
        src_attn = target
    if src_attn is None and cfg is None:
        return None

    # ---- Idempotency ----
    def _already_wrapped(obj: Any) -> bool:
        if obj is None:
            return False
        if isinstance(obj, AdaptiveSpanAttention):
            return True
        if "AdaptiveSpan" in obj.__class__.__name__:
            return True
        inner = getattr(obj, "inner", None)
        wrapped = getattr(obj, "_wrapped_attention", None)
        return isinstance(inner, AdaptiveSpanAttention) or isinstance(
            wrapped, AdaptiveSpanAttention
        )

    if _already_wrapped(src_attn):
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        if _already_wrapped(getattr(target, attr, None)):
            return None

    # ---- Skip DeltaNet/LinearAttention ----
    if src_attn is not None:
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

    # ---- Extract dims ----
    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    rank = getattr(comp_config, "adaptive_span_rank", kwargs.pop("rank", 32))
    dropout = getattr(comp_config, "dropout", kwargs.get("dropout", 0.1))
    max_span = getattr(comp_config, "max_span", kwargs.get("max_span", 128))

    is_nested_layer = (
        hasattr(target, "attention")
        and hasattr(target.attention, "output")
        and hasattr(target.attention.output, "dense")
    ) or (hasattr(target, "output") and hasattr(target.output, "dense"))

    try:
        new_block = AdaptiveSpanAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout,
            max_span=max_span,
            pretrained_attn=src_attn,
            is_nested_encoder=(True if is_nested_layer else kwargs.get("is_nested_encoder")),
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception:
        logger.exception("AdaptiveSpanAttention construction failed.")
        return None

    # ---- Splice ----
    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
            else:
                target.attention = new_block
        elif hasattr(target, "self"):
            target.self = new_block
        elif hasattr(target, "attn"):
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block

    return new_block