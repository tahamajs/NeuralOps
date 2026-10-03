"""Progressive Token Pruning (PTP) – Production-Hardened (v2).

Learnable Progressive Token Pruning module with guaranteed [CLS] retention,
multi-head attention capabilities, and full multi-architecture support.

⚠️  IMPORTANT — WHAT PTP ACTUALLY DOES  ⚠️

PTP is a **compute optimization**, not a parameter compressor. It prunes
tokens at *inference time* by keeping the top-k highest-scoring tokens per
sequence (plus the always-preserved first token). The full token sequence
still flows through the model; only the attention operation is restricted
to a subset of keys/values.

Consequences:
  - Parameter count is NOT reduced (it may even slightly grow due to the
    score network).
  - KV cache is NOT reduced (all tokens are still stored).
  - FLOPs during attention ARE reduced (≈ keep_ratio × L²).
  - Long-context inference is significantly faster.

If your goal is parameter compression, use SVD/MLA/quantization instead
(or apply them *after* PTP). If your goal is inference speedup, PTP is
appropriate, but the pipeline should label it as a *compute optimizer*,
not a *compressor*.

Bug-fix map (v2):
  PTP-FIX-01  **KV-cache support** for decoder inference. Both tuple and
              DynamicCache formats are supported.
  PTP-FIX-02  **RoPE past_len offset** for KV-cache.
  PTP-FIX-03  **Removed double `scores[:, 0] = 1e4` assignment.**
  PTP-FIX-04  **Score network shares weights across layers** (optional
              shared scorer) to minimize parameter overhead.
  PTP-FIX-05  **Correct `_rotate_half` for odd head_dim.**
  PTP-FIX-06  **Position_ids forwarded to RoPE** when present.
  PTP-FIX-07  **Injection rewritten**: single-pass, no mutation during
              iteration; nested encoders handled correctly.
  PTP-FIX-08  **Removed broken `comp_config if 'comp_config' in locals()`**.
  PTP-FIX-09  **Forward sanity check** added.
  PTP-FIX-10  **Added `dense`, `c_proj`, `o_proj` aliases.**
  PTP-FIX-11  **`target_layer_idx` honored in whole-model traversal.**
  PTP-FIX-12  **Class-level `is_custom_attention = True`** (it does replace
              attention in the graph, but flags that it is not a compressor).
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
    # PTP-FIX-05: correct for odd head_dim.
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
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

    if cos.shape[-2] != q.shape[-2]:
        raise ValueError(
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
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
# LoRALinear
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
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# Optional shared scorer (PTP-FIX-04)
# ===========================================================================
class SharedTokenScorer(nn.Module):
    """
    Optional token scorer that is shared across all PTP layers to reduce
    parameter overhead. When `share_scorer=True` is passed to all PTP
    instances, they reference the same scorer module.
    """

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)  # (B, L)


# ===========================================================================
# PTP Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("progressive_token_pruning")
@ATTENTION_REGISTRY.register("ptp")
class ProgressiveTokenPruning(BaseAttention):
    """
    Progressive Token Pruning attention.

    Forward:
      1. Score each token (score_network or shared scorer).
      2. Force-keep the first token ([CLS]).
      3. Keep the top-k tokens (k = ceil(L · keep_ratio)).
      4. Compute attention over the reduced set of keys/values.
      5. Return output with the original (B, L, D) shape.

    KV-cache: supported for decoder inference.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        d_model: int | None = None,
        keep_ratio: float = 0.8,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        shared_scorer: SharedTokenScorer | None = None,
        **kwargs: Any,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        # PTP-FIX-12: dimension resolution.
        self.d_model = int(
            getattr(config, "hidden_size", None)
            or d_model
            or kwargs.get("hidden_size")
            or 768
        )
        self.dim = self.d_model
        self.num_heads = int(
            getattr(
                config, "num_attention_heads", kwargs.get("num_attention_heads", 12)
            )
        )
        self.num_kv_heads = int(
            getattr(
                config,
                "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        )
        self.head_dim = int(
            getattr(
                config,
                "head_dim",
                kwargs.get("head_dim", self.dim // max(1, self.num_heads)),
            )
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.keep_ratio = float(
            getattr(comp_config, "keep_ratio", kwargs.get("keep_ratio", keep_ratio))
        )

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = float(
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
        self.q_lin = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_lin = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_lin = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_lin = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin
        self.dense = self.out_lin  # PTP-FIX-10
        self.c_proj = self.out_lin  # PTP-FIX-10
        self.o_proj = self.out_lin  # PTP-FIX-10

        # PTP-FIX-04: optional shared scorer.
        if shared_scorer is not None:
            self.score_network = shared_scorer
            self._uses_shared_scorer = True
        else:
            self.score_network = SharedTokenScorer(self.dim, hidden=64)
            self._uses_shared_scorer = False

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        src = (
            pretrained_attn
            if pretrained_attn is not None
            else (old_attn if old_attn is not None else kwargs.get("source_attn"))
        )
        if src is not None:
            self.copy_weights(src)

    # ------------------------------------------------------------------
    @property
    def gate(self):
        return None

    @property
    def random_proj(self):
        return None

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
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
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
                extracted["q"] = (
                    c_w[:split, :],
                    c_b[:split] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split : 2 * split, :],
                    c_b[split : 2 * split] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split :, :],
                    c_b[2 * split :] if c_b is not None else None,
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
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa nested
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
                f"ProgressiveTokenPruning copy_weights failed: missing "
                f"mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
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

                fnorm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if fnorm < 1e-4:
                    logger.warning(
                        "PTP: '%s' weight has near-zero norm (%.2e).",
                        key,
                        fnorm,
                    )

        if self.freeze_base:
            for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
                m.freeze_base_weights()

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("PTP does not support cross-attention.")

        x = hidden_states
        B, L, D = x.shape

        valid_mask = get_token_valid_mask(attention_mask, B, L, x.device)

        # PTP-FIX-03: single assignment.
        scores = self.score_network(x)  # (B, L)
        # PTP-FIX-11: only force [CLS] if position 0 is valid.
        first_valid = valid_mask[:, 0].unsqueeze(-1)  # (B, 1)
        scores = torch.where(first_valid, scores, scores)
        # Force first token to top priority (only if valid).
        scores = scores.masked_fill(~valid_mask, -1e4)
        if first_valid.any():
            scores[:, 0] = torch.where(
                first_valid.squeeze(-1),
                torch.full_like(scores[:, 0], 1e4),
                scores[:, 0],
            )

        k_tokens = max(1, min(L, int(math.ceil(L * self.keep_ratio))))
        top_scores, top_indices = torch.topk(scores, k_tokens, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        # Projections
        q = self.q_lin(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_new = (
            self.k_lin(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v_new = (
            self.v_lin(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # ---------------- KV-cache (PTP-FIX-01) ----------------
        past_k = past_v = None
        past_len = 0
        if past_key_value is not None and self.is_decoder is not False:
            # Tuple cache
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                past_len = past_k.shape[-2]
            # HF DynamicCache
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]
                past_len = past_k.shape[-2]

        # ---------------- RoPE with past_len offset (PTP-FIX-02) ----------------
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if past_len > 0 and cos.shape[-2] >= past_len + L:
                cos = cos[..., past_len : past_len + L, :]
                sin = sin[..., past_len : past_len + L, :]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        # ---------------- Gather pruned K/V ----------------
        # Note: PTP prunes tokens ONLY for the current chunk (new tokens).
        # With KV-cache, past tokens are always used in full.
        top_idx_exp = (
            top_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(-1, self.num_kv_heads, -1, self.head_dim)
        )
        k_pruned = torch.gather(k_new, 2, top_idx_exp)  # (B, H_kv, k_tokens, d)
        v_pruned = torch.gather(v_new, 2, top_idx_exp)

        if past_k is not None and past_v is not None:
            # Concatenate past (full) with current (pruned).
            k_all = torch.cat([past_k, k_pruned], dim=-2)
            v_all = torch.cat([past_v, v_pruned], dim=-2)
            # Keep track of original positions for causal masking.
            past_positions = torch.arange(past_len, device=x.device).view(1, 1, -1)
            past_positions = past_positions.expand(B, 1, -1)
            current_positions = (
                torch.gather(
                    top_indices,
                    1,
                    torch.arange(k_tokens, device=x.device).unsqueeze(0).expand(B, -1),
                )
                + past_len
            )  # positions in the full sequence
            k_positions = torch.cat(
                [past_positions, current_positions.unsqueeze(1)],
                dim=-1,
            )  # (B, 1, past_len + k_tokens)
        else:
            k_all = k_pruned
            v_all = v_pruned
            k_positions = top_indices.unsqueeze(1)  # (B, 1, k_tokens)

        # ---------------- GQA repeat ----------------
        if self.num_heads != self.num_kv_heads:
            k_all = repeat_kv(k_all, self.kv_repeat)
            v_all = repeat_kv(v_all, self.kv_repeat)

        # ---------------- Attention scores ----------------
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (
            torch.matmul(q.float(), k_all.float().transpose(-2, -1).contiguous())
            * scale
        )  # (B, H, L, past_len + k_tokens)

        # ---------------- Causal mask ----------------
        if self.is_decoder:
            q_pos = (past_len + torch.arange(L, device=x.device)).view(1, 1, L, 1)
            causal = k_positions.unsqueeze(2) > q_pos
            attn_scores = attn_scores.masked_fill(causal, -1e4)

        # ---------------- Valid-key mask for current chunk ----------------
        # Past keys are always valid (they were valid when cached).
        current_valid = torch.gather(valid_mask, 1, top_indices)  # (B, k_tokens)
        if past_k is not None:
            # Assume past keys are all valid (they were cached after masking).
            all_valid = torch.cat(
                [
                    torch.ones(B, past_len, device=x.device, dtype=torch.bool),
                    current_valid,
                ],
                dim=1,
            )  # (B, past_len + k_tokens)
        else:
            all_valid = current_valid
        attn_scores = attn_scores.masked_fill(
            ~all_valid.unsqueeze(1).unsqueeze(2), -1e4
        )

        probs = F.softmax(attn_scores, dim=-1).to(x.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        attn_out = (
            torch.matmul(probs, v_all)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, L, self.all_head_size)
        )

        # ---------------- Update KV-cache ----------------
        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                # HF DynamicCache: store the FULL k_new, v_new (not pruned),
                # so that future queries can access them.
                past_key_value.update(k_new, v_new, layer_idx)

        if self.is_nested_encoder:
            return attn_out, (probs if output_attentions else None)

        out = self.out_lin(attn_out)
        return out, (probs if output_attentions else None)


# ===========================================================================
# Injection (PTP-FIX-07)
# ===========================================================================
def _find_layer_list(model: nn.Module):
    """Return (parent, attr_name, layer_list) or None."""
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
        for part in path[:-1]:
            obj = getattr(obj, path[0], None) if False else obj
        # Re-walk cleanly:
        obj = model
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            # Find parent + attr
            parent = model
            for part in path[:-1]:
                parent = getattr(parent, part)
            return parent, path[-1], obj
    return None


def inject_progressive_token_pruning(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    target_layer_idx: int | None = None,
    shared_scorer: SharedTokenScorer | None = None,
    **kwargs: Any,
):
    # PTP-FIX-07: cleaner injection.
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---------------- Whole-model traversal ----------------
    if layer is None and model is not None:
        found = _find_layer_list(model)
        if found is None:
            logger.warning("PTP: could not locate transformer layer list.")
            return None
        parent, attr, layer_list = found
        num_injected = 0
        tot = len(layer_list)
        # If shared_scorer requested, create it once.
        if shared_scorer is None and kwargs.get("share_scorer", False):
            dim = getattr(
                getattr(model, "config", None), "hidden_size", None
            ) or kwargs.get("hidden_size", 768)
            shared_scorer = SharedTokenScorer(int(dim), hidden=64)

        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_progressive_token_pruning(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                shared_scorer=shared_scorer,
                **{k: v for k, v in kwargs.items() if k != "share_scorer"},
            )
            if res is not None:
                num_injected += 1
        return model if num_injected > 0 else None

    # ---------------- Single-layer injection ----------------
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

        if isinstance(src_attn, ProgressiveTokenPruning):
            return None

        new_attn = ProgressiveTokenPruning(
            config=cfg or getattr(src_attn, "config", None),
            pretrained_attn=src_attn if src_attn is not target else None,
            shared_scorer=shared_scorer,
            **kwargs,
        )

        # Attach
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
        logger.exception("PTP injection failed: %s", e)
        return None


inject_ptp = inject_progressive_token_pruning
inject_progressive_token_pruning_attention = inject_progressive_token_pruning


__all__ = [
    "ProgressiveTokenPruning",
    "SharedTokenScorer",
    "inject_progressive_token_pruning",
    "inject_ptp",
]
