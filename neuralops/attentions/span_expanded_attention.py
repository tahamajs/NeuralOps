"""Span-Expanded Attention with HyLoRA – Production-Hardened.

Extends attention receptive fields via fixed/cached key-value span expansion buffers,
complemented by parallel low-rank adaptation (HyLoRA) for parameter-efficient adaptation.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
- **Streaming KV Cache with Sliding Window + Sink Tokens** (StreamingLLM-style)
- **RoPE position offset for incremental decoding**
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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

    # Align the time dimension of cos/sin with q
    if cos.shape[-2] != q.shape[-2]:
        # If they don't match, truncate or broadcast depending on size
        if cos.shape[-2] >= q.shape[-2]:
            cos = cos[..., : q.shape[-2], :]
            sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


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
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
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
            lora_out = (
                F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            )
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("span_expanded")
@ATTENTION_REGISTRY.register("span_expanded_attention")
class SpanExpandedAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: int | None = None,
        expansion_span: int = 1024,
        num_sink_tokens: int = 4,
        use_lora: bool = True,
        lora_rank: int = 8,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 64)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.hidden_size: int = getattr(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        )
        self.dim = self.hidden_size
        self.num_heads: int = getattr(
            config, "num_attention_heads", kwargs.get("num_attention_heads", 12)
        )
        self.num_kv_heads: int = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim: int = getattr(
            config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim

        # ---- Span expansion configuration ----
        self.expansion_span: int = int(expansion_span) if expansion_span else 1024
        # Number of leading "sink" tokens to always preserve in the cache.
        # StreamingLLM (Xiao et al. 2024) shows the first few tokens act as
        # attention sinks and must be kept for long-context stability.
        self.num_sink_tokens: int = max(
            0, min(int(num_sink_tokens), self.expansion_span)
        )

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = (
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
            if config
            else lora_rank
        )
        self.lora_alpha = (
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Projections
        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else kwargs.get("source_attn")
        )
        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Weight copying (unchanged, production-hardened)
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        (c_proj_mod.__class__.__name__ == "Conv1D")
                        or (base_o.__class__.__name__ == "Conv1D")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules (GPT-2 alt, Qwen, LLaMA, DistilBERT)
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"SpanExpandedAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    else:
                        logger.warning(
                            "SpanExpandedAttention copy_weights: missing %s, using initialized weights",
                            proj_key,
                        )

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (
                    w.shape[0] < target_mod.weight.shape[0]
                    and target_mod.weight.shape[0] % w.shape[0] == 0
                ):
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
                    elif (
                        b.shape[0] < target_mod.bias.shape[0]
                        and target_mod.bias.shape[0] % b.shape[0] == 0
                    ):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(
                        "SpanExpandedAttention copy_weights: %s has near-zero norm (%.2e)",
                        proj_key,
                        fnorm,
                    )

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Cache management helpers
    # ------------------------------------------------------------------
    def _get_past_kv(
        self, past_key_value: Any
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        """Extract (past_k, past_v, past_len) from various cache formats."""
        if past_key_value is None:
            return None, None, 0

        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            past_k, past_v = past_key_value[0], past_key_value[1]
            return past_k, past_v, past_k.shape[-2]

        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            past_k = past_key_value.key_cache[0]
            past_v = (
                past_key_value.value_cache[0]
                if hasattr(past_key_value, "value_cache")
                else None
            )
            return past_k, past_v, past_k.shape[-2]

        return None, None, 0

    def _truncate_cache_with_sinks(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sliding-window truncation while preserving the first N sink tokens.

        StreamingLLM (Xiao et al., 2024): the first few tokens accumulate a
        disproportionate amount of attention and act as "attention sinks".
        Dropping them causes severe perplexity degradation in long contexts.
        We therefore always preserve the first `num_sink_tokens` entries.
        """
        span = self.expansion_span
        if span <= 0 or k.shape[-2] <= span:
            return k, v

        sink = self.num_sink_tokens
        if sink <= 0:
            # Pure rolling cache, no sinks preserved
            return k[..., -span:, :], v[..., -span:, :]

        if span <= sink:
            # Pathological config: keep only sinks, drop the rest
            return k[..., :span, :], v[..., :span, :]

        # Keep first `sink` tokens + most recent `span - sink` tokens
        sink_k = k[..., :sink, :]
        sink_v = v[..., :sink, :]
        recent_k = k[..., -(span - sink) :, :]
        recent_v = v[..., -(span - sink) :, :]
        return torch.cat([sink_k, recent_k], dim=-2), torch.cat(
            [sink_v, recent_v], dim=-2
        )

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
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None
    ]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        # ---- QKV projections ----
        q = self.q_proj(hidden_states).view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, Dh)
            .permute(0, 2, 1, 3)
        )

        # ---- Extract past cache ----
        past_k, past_v, past_len = self._get_past_kv(past_key_value)

        # ---- RoPE with past_len offset ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos_curr = (
                cos[..., past_len : past_len + L, :]
                if cos.shape[-2] >= past_len + L
                else cos
            )
            sin_curr = (
                sin[..., past_len : past_len + L, :]
                if sin.shape[-2] >= past_len + L
                else sin
            )
            q, k = apply_rotary_pos_emb(q, k, cos_curr, sin_curr)

        # ---- GQA repeat (before cache concat, so cache stays compact) ----
        # Note: If using GQA, we store the compact KV in cache and expand on use.
        # Here we expand first, which is simpler but uses more cache memory.
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            # Re-expand past cache if it was stored compact
            if past_k is not None and past_k.shape[1] != self.num_heads:
                past_k = repeat_kv(past_k, repeats)
                past_v = repeat_kv(past_v, repeats)
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        # ---- Concatenate with past cache ----
        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # ---- Sliding-window truncation with sink preservation ----
        k, v = self._truncate_cache_with_sinks(k, v)

        # ---- Present cache for next step ----
        # Store compact (un-repeated) KV if GQA is used to save memory:
        if self.num_heads != self.num_kv_heads and k.shape[1] == self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            # Take every `repeats`-th head to recover compact form
            # (repeat_kv duplicates contiguously, so index 0, repeats, 2*repeats, ...)
            compact_k = k[:, ::repeats, :, :].contiguous()
            compact_v = v[:, ::repeats, :, :].contiguous()
            present_key_value = (compact_k, compact_v)
        else:
            present_key_value = (k, v)

        scale = Dh**-0.5

        # ---- Fast path: PyTorch SDPA ----
        if attention_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                # is_causal only meaningful when processing > 1 token at once.
                # When past cache is present and we process 1 token, no causal mask needed.
                _is_causal = bool(self.is_decoder) and L > 1 and past_len == 0
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _sdpa_out = (
                    _sdpa_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
                _sdpa_out = torch.nan_to_num(
                    _sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4
                )
                out = (
                    self.out_proj(_sdpa_out)
                    if not self.is_nested_encoder
                    else _sdpa_out
                )
                return out, None, present_key_value
            except Exception:
                pass  # fall through to manual path

        # ---- Manual path ----
        scores = (
            torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        )

        # Causal masking: query i (position past_len + i) attends to keys 0..(past_len + i)
        is_causal = self.is_decoder or kwargs.get("is_causal", False)
        if is_causal and L > 1:
            kv_len = k.shape[-2]
            causal_mask = torch.triu(
                torch.full((L, kv_len), -1e4, device=scores.device),
                diagonal=1 + past_len,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # External attention mask (padding etc.)
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask)

        attn = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Head mask
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                attn = attn * head_mask.view(1, self.num_heads, 1, 1).to(attn.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                attn = attn * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(attn.dtype)
            elif head_mask.dim() == 4:
                attn = attn * head_mask.to(attn.dtype)

        context = (
            torch.matmul(attn, v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, self.all_head_size)
        )

        if self.is_nested_encoder:
            return context, (attn if output_attentions else None), present_key_value

        out = self.out_proj(context)
        return out, (attn if output_attentions else None), present_key_value


# ----------------------------------------------------------------------
# Injection helper (unchanged)
# ----------------------------------------------------------------------
def inject_span_expanded(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    target_layer_idx: int | None = None,
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
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
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
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_span_expanded(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    try:
        if layer is not None:
            config = model_config or getattr(layer, "config", None)
            target = old_attn
            if target is None:
                for attr in ("attention", "self_attn", "attn", "self"):
                    if hasattr(layer, attr):
                        cand = getattr(layer, attr)
                        if isinstance(cand, nn.Module) and cand is not layer:
                            target = cand
                            break
            if target is None:
                target = layer

            new_attn = SpanExpandedAttention(
                config=config,
                pretrained_attn=target if target is not layer else None,
                **kwargs,
            )
            for attr in ("self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return layer
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
                return layer
            layer.attention = new_attn
            return layer

        if model is None:
            return None

        replaced = 0
        for name, module in model.named_modules():
            is_attn = False
            if (
                hasattr(module, "q_lin")
                or hasattr(module, "query")
                or hasattr(module, "q_proj")
                or hasattr(module, "c_attn")
            ):
                is_attn = True

            if is_attn:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model

                new_attn = SpanExpandedAttention(
                    config=getattr(model, "config", None),
                    pretrained_attn=module,
                    **kwargs,
                )
                setattr(parent, child_name, new_attn)
                replaced += 1
                if target_layer_idx is not None and replaced > target_layer_idx:
                    break
        return model if replaced > 0 else None
    except Exception:
        return None


inject_span_expanded_attention = inject_span_expanded
