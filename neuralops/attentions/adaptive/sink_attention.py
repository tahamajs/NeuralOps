"""
SinkAttention — StreamingLLM-style attention with sink tokens and rolling KV-cache.

Implements the "Attention Sinks" mechanism from:
    Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
    (https://arxiv.org/abs/2309.17453)

═══════════════════════════════════════════════════════════════════════════
WHAT THIS MODULE DOES
═══════════════════════════════════════════════════════════════════════════

StreamingLLM allows a decoder to process sequences of UNBOUNDED length with
a BOUNDED KV-cache, by exploiting the empirical observation that a small
number of initial tokens ("attention sinks") absorb most of the attention
mass, and the rest is concentrated on a recent window.

Cache policy (per forward step):
    cache = [first `num_sink_tokens` tokens]  ++  [last (window_size - num_sink) tokens]

This gives O(window_size) memory per layer, regardless of total sequence
length. It works because:
  1. Sink tokens provide a stable "attention dump" for the softmax.
  2. Recent tokens provide local context.
  3. Middle tokens are typically low-information and can be evicted.

═══════════════════════════════════════════════════════════════════════════
FIX SUMMARY vs DRAFT VERSION
═══════════════════════════════════════════════════════════════════════════

🔴 CRITICAL
  FIX-01  Full attention implemented — Q/K/V projections, QK^T, softmax, attn·V.
  FIX-02  Output has same shape as input (residual connection safe).
  FIX-03  Pruning is applied to K/V cache, NOT to hidden_states.
  FIX-04  KV-cache supported (tuple + HF DynamicCache) with position tracking.

🟡 PERFORMANCE
  FIX-05  SDPA fast-path when no pruning is needed.
  FIX-06  GQA support via repeat_kv.

🟢 ARCHITECTURE
  FIX-07  Full multi-architecture `copy_weights` (BERT/RoBERTa/DistilBERT/
           GPT-2/LLaMA/Qwen/T5/DeBERTa).
  FIX-08  Absolute position tracking — pruned cache keeps original positions
           for correct causal masking.
  FIX-09  RoPE applied at insertion time (positions preserved across pruning).

═══════════════════════════════════════════════════════════════════════════
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
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE helper (always-slice semantics)
# ===========================================================================
def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. cos/sin are ALWAYS sliced to match the current q/k length."""
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

    L = q.shape[-2]
    S = cos.shape[-2]
    if S != L:
        if S >= past_len + L:
            cos = cos[..., past_len : past_len + L, :]
            sin = sin[..., past_len : past_len + L, :]
        elif S >= L:
            cos = cos[..., :L, :]
            sin = sin[..., :L, :]
        else:
            raise ValueError(
                f"SinkAttention RoPE: cos length {S} too short for q length {L} "
                f"(past_len={past_len})."
            )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
        k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear (shared projection primitive)
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
# SinkCache — a lightweight tuple-like cache that stores (k, v, positions)
# ===========================================================================
class SinkCache:
    """
    Custom cache container used by SinkAttention.

    Stores per-layer: keys, values, and absolute position indices.

    Attributes:
        keys:    list[Tensor] of shape (B, H_kv, N_k, d) per layer
        values:  list[Tensor] of shape (B, H_kv, N_k, d) per layer
        positions: list[Tensor] of shape (B, N_k) per layer
        seen_tokens: total number of tokens observed so far (across all steps)
    """

    def __init__(self) -> None:
        self.keys: list[torch.Tensor | None] = []
        self.values: list[torch.Tensor | None] = []
        self.positions: list[torch.Tensor | None] = []
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
        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)
            self.positions.append(None)

        if self.keys[layer_idx] is None:
            k_all = key_states
            v_all = value_states
            pos_all = positions
        else:
            k_all = torch.cat([self.keys[layer_idx], key_states], dim=-2)
            v_all = torch.cat([self.values[layer_idx], value_states], dim=-2)
            pos_all = torch.cat([self.positions[layer_idx], positions], dim=-1)

        self.keys[layer_idx] = k_all
        self.values[layer_idx] = v_all
        self.positions[layer_idx] = pos_all
        return k_all, v_all, pos_all

    # Tuple-cache compatibility (so external code can use it like a tuple)
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


# ===========================================================================
# SinkAttention
# ===========================================================================
@ATTENTION_REGISTRY.register("sink_attention")
class SinkAttention(BaseAttention):
    """
    StreamingLLM-style attention with sink tokens + rolling KV-cache.

    Key design points:
      - The FULL sequence of hidden_states is used for Q (query).
      - K/V are accumulated in the cache, then pruned to:
            [first `num_sink_tokens`] ++ [last `window_size - num_sink_tokens`]
      - Absolute positions are preserved across pruning (for causal masking).
      - Works for both bidirectional (encoder) and causal (decoder) models.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        num_sink_tokens: int = 4,
        window_size: int = 4096,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        # -------- Dimensions --------
        self.d_model = int(
            getattr(config, "hidden_size", None) or d_model
            or kwargs.get("hidden_size") or 768
        )
        self.dim = self.d_model
        self.num_heads = int(
            getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        )
        self.num_kv_heads = int(
            getattr(
                config, "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        )
        self.head_dim = int(
            getattr(
                config, "head_dim",
                kwargs.get("head_dim", self.dim // max(1, self.num_heads)),
            )
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # -------- Streaming config --------
        self.num_sink_tokens = int(
            getattr(comp_config, "num_sink_tokens",
                    kwargs.get("num_sink_tokens", num_sink_tokens))
            if comp_config else num_sink_tokens
        )
        self.window_size = int(
            getattr(comp_config, "window_size", kwargs.get("window_size", window_size))
            if comp_config else window_size
        )
        if self.num_sink_tokens < 1:
            raise ValueError("num_sink_tokens must be >= 1.")
        if self.window_size < self.num_sink_tokens + 1:
            raise ValueError("window_size must be > num_sink_tokens.")

        # -------- Decoder / encoder --------
        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # -------- LoRA --------
        self.lora_rank = int(
            getattr(config, "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            if config else 16
        )
        self.lora_alpha = float(
            getattr(config, "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
            if config else 32.0
        )
        self.freeze_base = self.lora_rank > 0
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # -------- Projections --------
        self.q_lin = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_lin = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_lin = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_lin = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # -------- Aliases for HF compatibility --------
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin
        self.dense = self.out_lin
        self.c_proj = self.out_lin
        self.o_proj = self.out_lin

        dropout_p = (
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        # -------- Weights --------
        src = (
            pretrained_attn if pretrained_attn is not None
            else (old_attn if old_attn is not None else kwargs.get("source_attn"))
        )
        if src is not None:
            self.copy_weights(src)

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_cache(cache: Any, layer_idx: int):
        """
        Returns (past_k, past_v, past_pos, cache_type) where cache_type is
        'tuple' | 'sink' | 'dynamic' | None.

        past_pos: (B, N_k) long tensor of absolute positions, or None if
                  the cache format does not track positions.
        """
        if cache is None:
            return None, None, None, None

        # Our custom SinkCache
        if isinstance(cache, SinkCache):
            if layer_idx >= len(cache.keys) or cache.keys[layer_idx] is None:
                return None, None, None, "sink"
            return (
                cache.keys[layer_idx],
                cache.values[layer_idx],
                cache.positions[layer_idx],
                "sink",
            )

        # Tuple: (k, v) or (k, v, positions)
        if isinstance(cache, (tuple, list)) and len(cache) >= 2:
            k, v = cache[0], cache[1]
            pos = cache[2] if len(cache) >= 3 else None
            if k is None:
                return None, None, None, "tuple"
            return k, v, pos, "tuple"

        # HF DynamicCache
        if hasattr(cache, "key_cache") and cache.key_cache:
            idx = min(layer_idx, len(cache.key_cache) - 1)
            return cache.key_cache[idx], cache.value_cache[idx], None, "dynamic"

        return None, None, None, None

    # ------------------------------------------------------------------
    # copy_weights
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
                hasattr(src, "self") and hasattr(src.self, "query")
                and hasattr(src, "output") and hasattr(src.output, "dense")
            )
            or (
                cls_name in (
                    "BertSelfAttention", "RobertaSelfAttention",
                    "XLMRobertaSelfAttention", "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
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

        # GPT-2 fused
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
                extracted["k"] = (
                    c_w[split:2 * split, :],
                    c_b[split:2 * split] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split:, :],
                    c_b[2 * split:] if c_b is not None else None,
                )

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

        # BERT/RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub, None)
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

        # Flat
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
                f"SinkAttention.copy_weights failed: missing "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.q_lin, "k": self.k_lin, "v": self.v_lin, "o": self.out_lin}

        with torch.no_grad():
            for key, target_mod in target_map.items():
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
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
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

        if self.freeze_base:
            for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
                m.freeze_base_weights()

    # ------------------------------------------------------------------
    # streaming pruning (applied to cached K/V, not hidden_states)
    # ------------------------------------------------------------------
    def _prune_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keep the first `num_sink_tokens` and the last `W - num_sink` tokens.

        k, v:       (B, H_kv, N_k, d)
        positions:  (B, N_k)
        """
        N_k = k.shape[-2]
        W = self.window_size
        S = self.num_sink_tokens

        if N_k <= W:
            return k, v, positions

        head = slice(0, S)
        tail = slice(N_k - (W - S), N_k)
        idx = torch.arange(N_k, device=k.device)
        keep_idx = torch.cat([idx[head], idx[tail]])

        k = k.index_select(-2, keep_idx)
        v = v.index_select(-2, keep_idx)
        positions = positions.index_select(-1, keep_idx)
        return k, v, positions

    # ------------------------------------------------------------------
    # forward
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
            raise NotImplementedError("SinkAttention does not support cross-attention.")

        x = hidden_states
        B, N, _ = x.shape
        device = x.device

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)

        # ---------- Read cache ----------
        past_k, past_v, past_pos, cache_type = self._read_cache(past_key_value, layer_idx)
        past_len = 0 if past_k is None else past_k.shape[-2]

        # ---------- Projections ----------
        q = self.q_lin(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_new = self.k_lin(x).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_new = self.v_lin(x).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # ---------- RoPE (uses absolute positions) ----------
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        # Absolute offset for RoPE is the number of tokens we have ever seen:
        # if past cache exists, offset = last_cached_pos + 1
        if past_pos is not None and past_pos.numel() > 0:
            offset = int(past_pos[:, -1].max().item()) + 1
        else:
            offset = past_len

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, past_len=offset)

        # ---------- Absolute positions of current tokens ----------
        new_positions = (
            torch.arange(offset, offset + N, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )

        # ---------- Concatenate past + new, then prune ----------
        if past_k is not None:
            k_all = torch.cat([past_k, k_new], dim=-2)
            v_all = torch.cat([past_v, v_new], dim=-2)
            if past_pos is not None:
                pos_all = torch.cat([past_pos, new_positions], dim=-1)
            else:
                # Fallback: reconstruct positions assuming contiguity.
                pos_all = torch.arange(k_all.shape[-2], device=device)
                pos_all = pos_all.unsqueeze(0).expand(B, -1)
        else:
            k_all = k_new
            v_all = v_new
            pos_all = new_positions

        # ---------- Prune (StreamingLLM magic happens HERE) ----------
        k_keep, v_keep, pos_keep = self._prune_kv(k_all, v_all, pos_all)

        # ---------- FIX-02: update cache and return it ----------
        updated_cache: Any = None
        if cache_type == "sink":
            # reuse the same SinkCache object
            # write back pruned tensors
            cache = past_key_value
            while len(cache.keys) <= layer_idx:
                cache.keys.append(None)
                cache.values.append(None)
                cache.positions.append(None)
            cache.keys[layer_idx] = k_keep
            cache.values[layer_idx] = v_keep
            cache.positions[layer_idx] = pos_keep
            cache.seen_tokens = max(cache.seen_tokens, int(pos_keep.max().item()) + 1)
            updated_cache = cache
        elif cache_type == "dynamic" or (cache_type is None and hasattr(past_key_value, "update")):
            # HF DynamicCache: update with full (unpruned) K/V.
            # (HF DynamicCache does not support positions; we store full K/V
            #  and prune only inside this module, so memory is not reduced.
            #  For memory savings, pass a SinkCache.)
            if past_key_value is not None and hasattr(past_key_value, "update"):
                past_key_value.update(k_new, v_new, layer_idx)
                updated_cache = past_key_value
        elif cache_type == "tuple" or (cache_type is None and past_key_value is not None):
            # Return a 3-tuple with positions.
            updated_cache = (k_keep, v_keep, pos_keep)

        # ---------- GQA expansion for attention ----------
        if self.num_heads != self.num_kv_heads:
            k_att = repeat_kv(k_keep, self.kv_repeat)
            v_att = repeat_kv(v_keep, self.kv_repeat)
        else:
            k_att = k_keep
            v_att = v_keep

        N_k = k_att.shape[-2]

        # ---------- SDPA fast-path (when we can't prune further) ----------
        use_sdpa = (
            not self.training
            and N_k <= self.window_size
            and past_k is None
            and N == N_k
        )
        if use_sdpa:
            if self.is_decoder:
                attn_out = F.scaled_dot_product_attention(
                    q, k_att, v_att, is_causal=True,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            else:
                attn_bias = None
                if attention_mask is not None:
                    attn_bias = torch.zeros(B, 1, N, N, device=device, dtype=q.dtype)
                    attn_bias = attn_bias.masked_fill(
                        ~valid_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                    )
                attn_out = F.scaled_dot_product_attention(
                    q, k_att, v_att, attn_mask=attn_bias,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
            attn_out = (
                attn_out.permute(0, 2, 1, 3).contiguous()
                .view(B, N, self.all_head_size)
            )
            if self.is_nested_encoder:
                return attn_out, None
            out = self.out_lin(attn_out)
            if updated_cache is not None or past_key_value is not None:
                return out, None, updated_cache
            return out, None

        # ---------- Manual attention ----------
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(
            q.float(), k_att.float().transpose(-2, -1).contiguous()
        ) * scale  # (B, H, N, N_k)

        # ---------- Causal mask using ABSOLUTE positions ----------
        if self.is_decoder:
            q_pos = new_positions.view(B, 1, N, 1)          # (B,1,N,1)
            k_pos = pos_keep.view(B, 1, 1, N_k)             # (B,1,1,N_k)
            causal = k_pos > q_pos
            scores = scores.masked_fill(causal, float("-inf"))

        # ---------- Padding mask (only for current-chunk tokens) ----------
        # Mark which key positions are valid.
        # Cache tokens are always valid (they were added when valid).
        if past_k is not None and past_pos is not None:
            # Current tokens' validity is in valid_mask; map to kept positions.
            # Simple approach: assume all cached tokens remain valid.
            all_valid = torch.ones(B, N_k, device=device, dtype=torch.bool)
        else:
            all_valid = valid_mask  # (B, N) == (B, N_k)

        # But we must apply padding only to the *current* tokens.
        # We'll build a validity tensor for k_keep positions:
        #   - past positions  → True
        #   - current valid positions → valid_mask
        if past_pos is not None and past_k is not None:
            # Determine, for each k_keep position, whether it belongs to past
            # or to the current chunk, and its validity.
            # new_positions contains the positions of current tokens.
            # past_pos contains past positions.
            # We use a scatter:
            max_pos = int(pos_keep.max().item()) + 1
            valid_lookup = torch.zeros(B, max_pos, device=device, dtype=torch.bool)
            # Past tokens always valid
            if past_pos is not None:
                valid_lookup.scatter_(1, past_pos, True)
            # Current tokens: valid_mask
            valid_lookup.scatter_(1, new_positions, valid_mask)
            all_valid = valid_lookup.gather(1, pos_keep)     # (B, N_k)

        scores = scores.masked_fill(
            ~all_valid.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        probs = F.softmax(scores, dim=-1).to(x.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # ---------- head_mask ----------
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        # ---------- Output ----------
        attn_out = (
            torch.matmul(probs, v_att)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, N, self.all_head_size)
        )

        attn_weights = probs if output_attentions else None

        if self.is_nested_encoder:
            return attn_out, attn_weights

        out = self.out_lin(attn_out)

        if updated_cache is not None or past_key_value is not None:
            return out, attn_weights, updated_cache
        return out, attn_weights


# ===========================================================================
# Injection helper
# ===========================================================================
def _find_layer_list(model: nn.Module):
    paths = [
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
    ]
    for path in paths:
        obj = model
        ok = True
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            parent = model
            for part in path[:-1]:
                parent = getattr(parent, part)
            return parent, path[-1], obj
    return None


def inject_sink_attention(
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

    # ---------- Whole-model traversal ----------
    if layer is None and model is not None:
        found = _find_layer_list(model)
        if found is None:
            logger.warning("SinkAttention: could not locate transformer layer list.")
            return None
        _parent, _attr, layer_list = found
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_sink_attention(
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

    # ---------- Single-layer injection ----------
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

        if isinstance(src_attn, SinkAttention):
            return None

        new_attn = SinkAttention(
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
        logger.exception("SinkAttention injection failed: %s", e)
        return None


inject_sink = inject_sink_attention


__all__ = [
    "SinkAttention",
    "SinkCache",
    "inject_sink_attention",
    "inject_sink",
]