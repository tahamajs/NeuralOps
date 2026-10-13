"""
Span-Expanded Attention with HyLoRA (StreamingLLM-style rolling KV-cache).

⚠️  ARCHITECTURAL HONESTY NOTE  ⚠️

Despite the name, the mechanism implemented here is **StreamingLLM
(Xiao et al. 2023)**, not the original Span-Expanded retrieval of
Amazon's paper. It uses a rolling KV-cache with:
    - fixed number of leading "sink" tokens (always kept)
    - sliding recent window (keeps the last `expansion_span - num_sink` tokens)

If you actually need the Amazon Span-Expanded mechanism (retrieval of
evicted blocks into the context), this module is NOT it.


"""

from __future__ import annotations
from neuralops.core.common import _find_layer_list

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

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
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# SinkCache (recommended cache container)
# ===========================================================================
class SinkCache:
    """
    Cache container used by SpanExpandedAttention.

    Attributes:
        keys:    list[Tensor] (B, H_kv, N_k, d) per layer
        values:  list[Tensor] (B, H_kv, N_k, d) per layer
        positions: list[Tensor] (B, N_k) absolute positions per layer
        seen_tokens: total tokens observed across all layers (absolute counter)
    """

    def __init__(self, num_layers: int = 0) -> None:
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.positions: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.keys) or self.keys[layer_idx] is None:
            return 0
        return int(self.keys[layer_idx].shape[-2])

    def get_max_length(self) -> int:
        return self.seen_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.keys):
            self.keys.extend([None] * (layer_idx - len(self.keys) + 1))
            self.values.extend([None] * (layer_idx - len(self.values) + 1))
            self.positions.extend([None] * (layer_idx - len(self.positions) + 1))

        k_old = self.keys[layer_idx]
        v_old = self.values[layer_idx]
        p_old = self.positions[layer_idx]

        if k_old is None:
            k_all = key_states
            v_all = value_states
            p_all = positions
        else:
            k_all = torch.cat([k_old, key_states], dim=-2)
            v_all = torch.cat([v_old, value_states], dim=-2)
            p_all = torch.cat([p_old, positions], dim=-1)

        self.keys[layer_idx] = k_all
        self.values[layer_idx] = v_all
        self.positions[layer_idx] = p_all
        return k_all, v_all, p_all

    # Tuple-like access for compatibility
    def __getitem__(self, idx: int):
        if idx == 0:
            return self.keys
        if idx == 1:
            return self.values
        if idx == 2:
            return self.positions
        raise IndexError(idx)

    def __len__(self) -> int:
        return 3


# DEPRECATED: Dynamic span expansion defaults to full sequence horizon without dynamic pruning signals.
# @ATTENTION_REGISTRY.register("span_expanded_attention")
class SpanExpandedAttention(BaseAttention):
    """DEPRECATED: Dynamic span expansion defaults to full sequence horizon without dynamic pruning signals.

    This module is deprecated. Use 'adaptive_span_attention' instead.
    """
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

        self.hidden_size: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
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

        default_span = 64
        self.expansion_span: int = int(kwargs.get("expansion_span", default_span if (not expansion_span or expansion_span == 1024) else expansion_span))
        self.num_sink_tokens: int = max(0, min(int(num_sink_tokens), self.expansion_span))

        if self.expansion_span <= self.num_sink_tokens:
            raise ValueError(
                f"expansion_span ({self.expansion_span}) must be > "
                f"num_sink_tokens ({self.num_sink_tokens})."
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

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # copy_weights (unchanged — production hardened)
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name in [
                    "BertSelfAttention", "RobertaSelfAttention",
                    "XLMRobertaSelfAttention", "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused c_attn
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
                        o_w, bias_o.data.clone() if bias_o is not None else None,
                    )

        # RoBERTa/BERT nested
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

        # Flat
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

        missing_after = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"SpanExpandedAttention copy_weights failed: missing "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj, "k": self.k_proj,
            "v": self.v_proj, "o": self.out_proj,
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
                    continue

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

                fnorm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if fnorm < 1e-4:
                    logger.warning(
                        "SpanExpandedAttention copy_weights: %s has near-zero "
                        "norm (%.2e).", proj_key, fnorm,
                    )

        if self.freeze_base:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _get_past_kv(
        self, past_key_value: Any, layer_idx: int = 0
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
        """
        Extract (past_k, past_v, past_positions, absolute_seen) from various
        cache formats.

        Returns:
            past_k:   (B, H_kv, N_k, d) or None
            past_v:   (B, H_kv, N_k, d) or None
            past_pos: (B, N_k) absolute positions or None
            seen:     total number of tokens ever seen (absolute counter)
        """
        if past_key_value is None:
            return None, None, None, 0

        # SinkCache
        if isinstance(past_key_value, SinkCache):
            if layer_idx >= len(past_key_value.keys) or past_key_value.keys[layer_idx] is None:
                return None, None, None, 0
            return (
                past_key_value.keys[layer_idx],
                past_key_value.values[layer_idx],
                past_key_value.positions[layer_idx],
                past_key_value.seen_tokens,
            )

        # Tuple
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_pos = past_key_value[2] if len(past_key_value) >= 3 else None
            if past_k is None:
                return None, None, None, 0
            seen = int(past_pos[:, -1].max().item()) + 1 if past_pos is not None else past_k.shape[-2]
            return past_k, past_v, past_pos, seen

        # HF DynamicCache
        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            idx = min(layer_idx, len(past_key_value.key_cache) - 1)
            past_k = past_key_value.key_cache[idx]
            past_v = past_key_value.value_cache[idx]
            # HF DynamicCache does not store positions; rely on len.
            seen = past_k.shape[-2]
            return past_k, past_v, None, seen

        return None, None, None, 0

    def _truncate_cache_with_sinks(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply sliding-window truncation while preserving the first N sink tokens.

        k, v:       (B, H_kv, N_k, d)
        positions:  (B, N_k) — absolute positions
        """
        span = self.expansion_span
        N_k = k.shape[-2]

        if span <= 0 or N_k <= span:
            return k, v, positions

        sink = self.num_sink_tokens
        if sink <= 0:
            keep = slice(N_k - span, N_k)
            return k[..., keep, :], v[..., keep, :], positions[..., keep]

        # Keep first `sink` + last `span - sink`
        sink_idx = torch.arange(0, sink, device=k.device)
        recent_start = N_k - (span - sink)
        recent_idx = torch.arange(recent_start, N_k, device=k.device)
        keep_idx = torch.cat([sink_idx, recent_idx])

        return (
            k.index_select(-2, keep_idx),
            v.index_select(-2, keep_idx),
            positions.index_select(-1, keep_idx),
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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "SpanExpandedAttention does not support cross-attention."
            )

        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device

        # ---- 1) QKV projections (compact, no GQA expansion yet) ----
        q = self.q_proj(hidden_states).view(B, L, H, Dh).permute(0, 2, 1, 3).contiguous()
        k_new = self.k_proj(hidden_states).view(
            B, L, self.num_kv_heads, Dh
        ).permute(0, 2, 1, 3).contiguous()
        v_new = self.v_proj(hidden_states).view(
            B, L, self.num_kv_heads, Dh
        ).permute(0, 2, 1, 3).contiguous()

        # ---- 2) Extract past cache (compact) ----
        past_k, past_v, past_pos, seen = self._get_past_kv(past_key_value, layer_idx)

        # ---- 3) FIX-01: absolute position offset ----
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            # Explicit positions (preferred)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            start_pos = int(position_ids[0, 0].item())
            new_positions = position_ids.long()
        elif past_pos is not None:
            start_pos = int(past_pos[:, -1].max().item()) + 1
            new_positions = (
                torch.arange(start_pos, start_pos + L, device=device)
                .unsqueeze(0).expand(B, -1)
            )
        else:
            # Fallback: use `seen` counter (from custom cache or HF cache length)
            # NOTE: for HF DynamicCache without position tracking this can be
            # correct only if no pruning was ever applied.
            start_pos = seen
            new_positions = (
                torch.arange(start_pos, start_pos + L, device=device)
                .unsqueeze(0).expand(B, -1)
            )
            if past_pos is None and past_k is not None and past_k.shape[-2] >= self.expansion_span:
                from neuralops.utils.logging_config import warn_once
                warn_once(
                    "span_expanded_cache_sat",
                    "SpanExpandedAttention: cache length has saturated at "
                    "expansion_span=%d but no position tracking is available. "
                    "RoPE positions may be incorrect. Provide position_ids or "
                    "use SinkCache.",
                    self.expansion_span,
                )

        # ---- 4) RoPE with absolute offset ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, past_len=start_pos)

        # ---- 5) FIX-03: concat + truncate on COMPACT KV ----
        if past_k is not None:
            k_full = torch.cat([past_k, k_new], dim=-2)
            v_full = torch.cat([past_v, v_new], dim=-2)
            if past_pos is not None:
                pos_full = torch.cat([past_pos, new_positions], dim=-1)
            else:
                pos_full = (
                    torch.arange(k_full.shape[-2], device=device)
                    .unsqueeze(0).expand(B, -1)
                )
        else:
            k_full = k_new
            v_full = v_new
            pos_full = new_positions

        # ---- 6) FIX-02: prefill vs decode ----
        is_prefill = L > 1

        if is_prefill:
            # Prefill: compute attention on the FULL sequence (causality-safe).
            attn_k = k_full
            attn_v = v_full
            attn_pos = pos_full
            # Cache the TRUNCATED version for future steps.
            cache_k, cache_v, cache_pos = self._truncate_cache_with_sinks(
                k_full, v_full, pos_full
            )
        else:
            # Decode: truncate first (memory-efficient), then attend.
            cache_k, cache_v, cache_pos = self._truncate_cache_with_sinks(
                k_full, v_full, pos_full
            )
            attn_k = cache_k
            attn_v = cache_v
            attn_pos = cache_pos

        # ---- 7) Prepare present_key_value ----
        if isinstance(past_key_value, SinkCache):
            past_key_value.keys[layer_idx if layer_idx < len(past_key_value.keys) else -1] = cache_k
            if layer_idx >= len(past_key_value.keys):
                past_key_value.keys.append(cache_k)
                past_key_value.values.append(cache_v)
                past_key_value.positions.append(cache_pos)
            else:
                past_key_value.keys[layer_idx] = cache_k
                past_key_value.values[layer_idx] = cache_v
                past_key_value.positions[layer_idx] = cache_pos
            past_key_value.seen_tokens = max(past_key_value.seen_tokens, start_pos + L)
            present_key_value = past_key_value
        elif hasattr(past_key_value, "update"):
            # HF DynamicCache: store FULL K/V (positions handled upstream).
            past_key_value.update(k_new, v_new, layer_idx)
            present_key_value = past_key_value
        else:
            # Tuple cache: return (k, v, positions)
            present_key_value = (cache_k, cache_v, cache_pos)

        # ---- 8) FIX-03: expand GQA just before attention ----
        if self.num_heads != self.num_kv_heads:
            attn_k_exp = repeat_kv(attn_k, self.kv_repeat)
            attn_v_exp = repeat_kv(attn_v, self.kv_repeat)
        else:
            attn_k_exp = attn_k
            attn_v_exp = attn_v

        N_k = attn_k_exp.shape[-2]

        # ---- 9) FIX-05: SDPA fast path (only when safe) ----
        use_sdpa = (
            attention_mask is None
            and head_mask is None
            and not output_attentions
            and not self.training
        )
        if use_sdpa:
            # is_causal only valid when we're at the "real" prefill start
            _is_causal = bool(self.is_decoder) and is_prefill and start_pos == 0
            try:
                sdpa_out = F.scaled_dot_product_attention(
                    q, attn_k_exp, attn_v_exp,
                    is_causal=_is_causal,
                    dropout_p=0.0,
                )
                sdpa_out = (
                    sdpa_out.permute(0, 2, 1, 3).contiguous()
                    .view(B, L, self.all_head_size)
                )
                sdpa_out = torch.nan_to_num(sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                if self.is_nested_encoder:
                    return sdpa_out, None, present_key_value
                out = self.out_proj(sdpa_out)
                return out, None, present_key_value
            except Exception:
                pass  # fall through

        # ---- 10) Manual attention ----
        scale = Dh ** -0.5
        scores = torch.matmul(
            q.float(), attn_k_exp.float().transpose(-2, -1).contiguous()
        ) * scale  # (B, H, L, N_k)

        # FIX-02: causal mask uses absolute positions of K and Q.
        if self.is_decoder:
            q_pos = new_positions.view(B, 1, L, 1)          # absolute
            k_pos = attn_pos.view(B, 1, 1, N_k)             # absolute
            causal = k_pos > q_pos
            scores = scores.masked_fill(causal, float("-inf"))

        # External mask (padding)
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask, is_causal=False)

        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # FIX-06: head_mask
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        context = (
            torch.matmul(probs, attn_v_exp)
            .permute(0, 2, 1, 3).contiguous()
            .view(B, L, self.all_head_size)
        )
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_weights = probs if output_attentions else None

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if output_attentions:
            return out, attn_weights, present_key_value
        return out, present_key_value


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
    if layer is None and model is not None:
        found = _find_layer_list(model)
        if found is None:
            logger.warning("SpanExpandedAttention: could not locate layer list.")
            return None
        _parent, _attr, layer_list = found
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
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
        return model if num_injected > 0 else None

    # Single-layer injection
    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src_attn = getattr(cand, "self", cand)
                        break
        if src_attn is None:
            src_attn = target

        if isinstance(src_attn, SpanExpandedAttention):
            return None

        new_attn = SpanExpandedAttention(
            config=cfg or getattr(src_attn, "config", None),
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            return new_attn
        if hasattr(target, "self_attn"):
            target.self_attn = new_attn
            return new_attn
        if hasattr(target, "attn"):
            target.attn = new_attn
            return new_attn
        if hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            return new_attn
        return None
    except Exception as e:
        logger.exception("SpanExpandedAttention injection failed: %s", e)
        return None


inject_span_expanded_attention = inject_span_expanded


__all__ = [
    "SpanExpandedAttention",
    "SinkCache",
    "inject_span_expanded",
    "inject_span_expanded_attention",
]