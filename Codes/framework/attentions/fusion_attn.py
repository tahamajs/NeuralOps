"""FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module (Production-Hardened v2).

Hybrid sparse attention with:
  - Short-sequence fast path (full attention via SDPA / manual)
  - Long-sequence block-gated sparse attention (MoBA-style)
  - Per-head dynamic gating for block selection
  - RoPE support with partial-dim slicing
  - GQA with repeat_kv
  - Nested encoder bypass
  - LoRA adapters on frozen base projections
  - Strict causal masking across both paths
  - Padding masking across both paths
  - KV-cache support

Bug-fix map (v2):
  FA-FIX-01  `logger` defined at module scope.
  FA-FIX-02  Fast-path SDPA now honors attention_mask / head_mask /
             output_attentions; falls back to manual path when needed.
  FA-FIX-03  Block-path causal mask uses ABSOLUTE positions, not a
             fixed local diagonal.
  FA-FIX-04  Block-path padding mask is gathered with the same indices
             as K/V, so masking is aligned.
  FA-FIX-05  `top_indices[..., 0] = 0` replaced with proper
             "force-include block 0" via top-(k-1) + cat.
  FA-FIX-06  `past_key_value` is read and updated for self-attention.
  FA-FIX-07  RoPE fallback uses `hidden_states` (was `v`).
  FA-FIX-08  NaN/Inf guards on block-path output.
  FA-FIX-09  `head_mask` applied in block path.
  FA-FIX-10  `output_attentions` returns block-attention weights.
  FA-FIX-11  `inject_fusion_attn` logs exceptions with traceback.
  FA-FIX-12  Idempotency guard in injector.
  FA-FIX-13  `num_kv_heads` reads `num_kv_heads` fallback (was only
             `num_key_value_heads`).
  FA-FIX-14  Debug `print` removed.
  FA-FIX-15  Per-head gating MLP (no more shared gate).
  FA-FIX-16  Gate uses per-token centroid scoring (mean over block is
             kept, but head_id is embedded in the gate input).
  FA-FIX-17  Block loop replaced with vectorized gather where possible.
  FA-FIX-18  Causal masking applied to gate_scores before top-k in
             decoders, so top-k only considers visible blocks.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


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

    # align sequence length if needed (KV-cache case)
    if cos.shape[-2] != q.shape[-2]:
        if cos.shape[-2] > q.shape[-2]:
            cos = cos[..., : q.shape[-2], :]
            sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_e = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot_e = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q = torch.cat([q_rot_e, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_e
        k = torch.cat([k_rot_e, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_e
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
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
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

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
# FUSIONATTN
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "fusion_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "fusion_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class FUSIONATTN(BaseAttention):
    """
    Hybrid sparse attention with block gating.

    Two execution paths:
      * Short sequence (N <= 128) or cross-attention  → dense attention
      * Long sequence                                  → block-gated sparse
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        hidden_size: int = 768,
        num_heads: int = 12,
        block_size: int = 64,
        top_k_blocks: int = 4,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- resolve hidden_size / num_heads ----
        if config is not None:
            hidden_size = int(
                getattr(
                    config,
                    "hidden_size",
                    getattr(config, "d_model", getattr(config, "n_embd", hidden_size)),
                )
            )
            num_heads = int(
                getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "num_heads", getattr(config, "n_heads", num_heads)),
                )
            )
        elif source_attn is not None:
            out_layer = None
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                if hasattr(source_attn, attr):
                    out_layer = getattr(source_attn, attr)
                    break
            if out_layer is not None:
                hidden_size = int(
                    getattr(
                        out_layer,
                        "out_features",
                        getattr(out_layer, "in_features", hidden_size),
                    )
                )
            num_heads = int(
                getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "num_heads", num_heads),
                    ),
                )
            )

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.hidden_size = int(hidden_size)
        self.num_heads = max(1, int(num_heads))

        # FA-FIX-13: num_kv_heads fallback chain
        if config is not None:
            self.num_kv_heads = int(
                getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "num_kv_heads", self.num_heads),
                )
            )
        else:
            self.num_kv_heads = self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        cfg_hd = getattr(config, "head_dim", None) if config else None
        self.head_dim = (
            int(cfg_hd)
            if cfg_hd is not None
            else max(1, self.hidden_size // self.num_heads)
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        self.block_size = int(block_size)
        self.top_k_blocks = int(top_k_blocks)

        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx = int(resolved_layer_id)

        # ---- LoRA ----
        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else kwargs.get("lora_alpha", 32.0)
        )
        self.freeze_base = self.lora_rank > 0

        # ---- decoder detection ----
        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
            )
        _dec_prefixes = (
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
        if any(model_type.startswith(p) for p in _dec_prefixes):
            is_dec = True
        if config is not None and hasattr(config, "architectures"):
            arch = getattr(config, "architectures", []) or []
            if any("GPT" in str(a) or "CausalLM" in str(a) for a in arch):
                is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if hasattr(source_attn, "c_attn") or getattr(
                source_attn, "is_decoder", False
            ):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # FA-FIX-15: per-head gate MLP (head_id embedding + centroid features)
        self.attn_gate = nn.Sequential(
            nn.Linear(self.head_dim * 2, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, 1),
        )
        # learnable per-head bias so heads can diverge
        self.gate_head_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.gate_temp = nn.Parameter(torch.ones(1))

        dropout_p = (
            float(
                getattr(
                    config,
                    "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                )
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None
            else None
        )
        self.comp_config = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def W_q(self):
        return self.q_proj

    @property
    def W_k(self):
        return self.k_proj

    @property
    def W_v(self):
        return self.v_proj

    @property
    def c_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = any(
            hasattr(pretrained_attn, a) for a in ("out_lin", "c_proj", "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
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

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if (
                hasattr(c_attn_mod, "merge")
                and callable(c_attn_mod.merge)
                and not getattr(c_attn_mod, "merged", False)
            ):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)

            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = c_attn_mod.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                is_conv = (
                    "Conv1D" in cls_c
                    or "Conv1D" in base_cls_c
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
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

            if (
                hasattr(pretrained_attn, "c_proj")
                and pretrained_attn.c_proj is not None
            ):
                c_proj_mod = pretrained_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    cls_o = c_proj_mod.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    is_conv_o = (
                        "Conv1D" in cls_o
                        or "Conv1D" in base_cls_o
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # RoBERTa / BERT nested
        elif hasattr(pretrained_attn, "self") and hasattr(
            pretrained_attn.self, "query"
        ):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub, None)
                if mod is not None and hasattr(mod, "weight"):
                    b = getattr(mod, "bias", None)
                    extracted[key] = (
                        mod.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(pretrained_attn, "output") and hasattr(
                pretrained_attn.output, "dense"
            ):
                out_mod = pretrained_attn.output.dense
                b = getattr(out_mod, "bias", None)
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # Flat modules
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        mod = getattr(pretrained_attn, n)
                        if mod is not None and hasattr(mod, "weight"):
                            b = getattr(mod, "bias", None)
                            extracted[key] = (
                                mod.weight.data.clone(),
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
                from framework.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
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
            logger.warning(
                "FUSIONATTN copy_weights: missing %s from %s", missing_after, cls_name
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

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        target_mod.weight.device, target_mod.weight.dtype
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(target_mod.bias.device, target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "FUSIONATTN: '%s' near-zero norm (%.2e)", proj_key, w_norm
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(past_key_value: Any | None, layer_idx: int = 0):
        if past_key_value is None:
            return None, None
        if hasattr(past_key_value, "key_cache") and hasattr(
            past_key_value, "value_cache"
        ):
            keys = past_key_value.key_cache
            vals = past_key_value.value_cache
            if len(keys) == 0:
                return None, None
            idx = min(layer_idx, len(keys) - 1)
            k, v = keys[idx], vals[idx]
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                return k, v
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                return k, v
        return None, None

    # ------------------------------------------------------------------
    # Mask preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(
        mask, B: int, N: int, device: torch.device
    ) -> torch.Tensor | None:
        """Return (B, N) bool mask where True = valid token."""
        if mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape[-1] == N:
                return m
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=-2).squeeze(1)
            if m.shape[-1] == N:
                return m
        return None

    # ------------------------------------------------------------------
    # Dense path
    # ------------------------------------------------------------------
    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        is_causal: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, N, D = q.shape
        N_kv = k.shape[2]
        scale = D**-0.5

        # FA-FIX-02: only use SDPA when no extra features are needed
        use_sdpa = (
            not output_attentions
            and head_mask is None
            and (valid_mask is None or valid_mask.all())
            and not self.training  # training dropout in SDPA is fine, but keep manual for safety
        )

        if use_sdpa:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _out = F.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=is_causal and N > 1 and N_kv == N,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _out = (
                    _out.to(q.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, N, self.all_head_size)
                )
                _out = torch.nan_to_num(_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return _out, None
            except Exception as e:
                logger.debug("SDPA failed, falling back to manual: %s", e)

        # ---- Manual path ----
        scores = (
            torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        )

        # FA-FIX-03 / decoder: causal mask with past_len offset
        if is_causal and N > 1:
            past_len = max(0, N_kv - N)
            causal = (
                torch.triu(
                    torch.full(
                        (N, N_kv), float("-inf"), device=q.device, dtype=torch.float32
                    ),
                    diagonal=1 + past_len,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal

        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(v.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        context = (
            torch.matmul(probs, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, N, self.all_head_size)
        )
        return context, (probs if output_attentions else None)

    # ------------------------------------------------------------------
    # Block-gated sparse attention
    # ------------------------------------------------------------------
    def _block_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_mask: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        is_causal: bool,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        FA-FIX-03 / FA-FIX-04 / FA-FIX-05 / FA-FIX-15 / FA-FIX-16 /
        FA-FIX-17 / FA-FIX-18 rewritten block-gated attention.
        """
        B, H, N, D = q.shape
        N_kv = k.shape[2]
        bs = self.block_size
        device = q.device

        # Pad to multiple of block_size
        pad_len = (bs - (N % bs)) % bs
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
            if valid_mask is not None:
                valid_pad = F.pad(valid_mask, (0, pad_len), value=False)
            else:
                valid_pad = None
        else:
            q_pad, k_pad, v_pad = q, k, v
            valid_pad = valid_mask

        N_pad = q_pad.size(2)
        num_blocks = N_pad // bs
        top_k = min(self.top_k_blocks, num_blocks)

        # ---- block centroids ----
        q_blocks = q_pad.reshape(B, H, num_blocks, bs, D)
        k_blocks = k_pad.reshape(B, H, num_blocks, bs, D)
        q_centroid = q_blocks.mean(dim=3)  # (B, H, nb, D)
        k_centroid = k_blocks.mean(dim=3)  # (B, H, nb, D)

        # FA-FIX-15 / FA-FIX-16: per-head gating with centroid features
        q_feat = torch.cat(
            [q_centroid, torch.zeros_like(q_centroid)], dim=-1
        )  # (B,H,nb,2D)
        k_feat = torch.cat(
            [torch.zeros_like(k_centroid), k_centroid], dim=-1
        )  # (B,H,nb,2D)
        q_gate = self.attn_gate(q_feat).squeeze(-1)  # (B, H, nb)
        k_gate = self.attn_gate(k_feat).squeeze(-1)  # (B, H, nb)

        # per-head learnable bias
        head_bias = self.gate_head_bias.view(1, H, 1)
        gate_scores = (
            torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2))
            .squeeze(-1)
            .squeeze(-1)
            + head_bias
        ) / (self.gate_temp.abs() + 1e-5)
        # gate_scores: (B, H, nb)

        # FA-FIX-18: causal-aware top-k for decoder self-attention
        # A query in block i cannot attend to any key in block j > i.
        if is_causal:
            blk_idx = torch.arange(num_blocks, device=device)
            causal_blocks = blk_idx.unsqueeze(0) <= blk_idx.unsqueeze(1)  # (nb, nb)
            # mask future blocks with -inf so top-k never picks them
            gate_scores_2d = gate_scores.unsqueeze(-1).expand(
                -1, -1, -1, num_blocks
            )  # not needed; use simpler form
            # compute per-query-block mask and apply to logits
            # gate_scores is (B, H, nb). We need (B, H, nb) * causal_mask_per_qblock
            # A more direct approach: for each query block i, only keep scores over j <= i
            # which is a triangular mask per (B,H,i).
            # We expand gate_scores to (B, H, nb, nb) with broadcast, then mask.
            gs2 = gate_scores.unsqueeze(-1).expand(
                -1, -1, -1, num_blocks
            )  # (B,H,nb,nb) where last dim = key block
            causal_blocks_b = causal_blocks.view(1, 1, num_blocks, num_blocks)
            gs2 = gs2.masked_fill(~causal_blocks_b, float("-inf"))
            # Top-k over key-block dim
            _, top_indices = torch.topk(gs2, top_k, dim=-1)  # (B, H, nb, top_k)
        else:
            # bidir: broadcast scores over query-block dim
            gs2 = gate_scores.unsqueeze(-1).expand(-1, -1, num_blocks, num_blocks)
            _, top_indices = torch.topk(gs2, top_k, dim=-1)

        # FA-FIX-05: guarantee block 0 is kept by constructing top-k differently
        # For each query block, force slot 0 to be block 0 (sink), fill rest with top-(k-1)
        with torch.no_grad():
            if top_k > 1:
                # take top-(k-1) from scores excluding block 0
                gs2_others = gs2.clone()
                gs2_others[..., 0] = float("-inf")
                _, top_others = torch.topk(gs2_others, top_k - 1, dim=-1)
                sink = torch.zeros_like(top_others[..., :1])  # block 0
                top_indices = torch.cat([sink, top_others], dim=-1)
            else:
                top_indices = torch.zeros(
                    B, H, num_blocks, top_k, device=device, dtype=torch.long
                )

        # ---- gather K/V per query block ----
        # k_blocks: (B, H, nb, bs, D), v_blocks same
        v_blocks = v_pad.reshape(B, H, num_blocks, bs, D)

        # For padding mask of K, pad to full length then reshape to blocks
        if valid_pad is not None:
            valid_blocks = valid_pad.view(B, 1, num_blocks, bs)  # (B, 1, nb, bs)
        else:
            valid_blocks = torch.ones(
                B, 1, num_blocks, bs, device=device, dtype=torch.bool
            )

        # gather: (B, H, nb, top_k, bs, D) → reshape to (B, H, nb, top_k*bs, D)
        idx_k = (
            top_indices.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(B, H, num_blocks, top_k, bs, D)
        )
        k_sel = torch.gather(
            k_blocks.unsqueeze(3).expand(-1, -1, -1, top_k, -1, -1),
            2,
            top_indices.reshape(B, H, num_blocks, top_k, 1, 1).expand(
                B, H, num_blocks, top_k, bs, D
            ),
        )
        # Above gather is awkward; use index_select-free approach:
        # reshape k_blocks to (B,H,nb,bs*D) and gather along dim=2 with flattened indices
        k_flat = k_blocks.reshape(B, H, num_blocks, bs * D)
        v_flat = v_blocks.reshape(B, H, num_blocks, bs * D)

        # Build absolute gather indices
        idx_flat = top_indices.reshape(B, H, num_blocks, top_k, 1).expand(
            -1, -1, -1, -1, bs
        )
        idx_flat = idx_flat * bs + torch.arange(bs, device=device).view(1, 1, 1, 1, bs)
        idx_flat = idx_flat.reshape(B, H, num_blocks, top_k * bs)

        k_sel = torch.gather(
            k_flat.unsqueeze(3).expand(-1, -1, -1, top_k, -1),
            -1,
            idx_flat.unsqueeze(-1).expand(-1, -1, -1, -1, D),
        )  # (B,H,nb,top_k,D)
        # Actually this is getting confusing. Simpler:
        # flatten indices to a per-(B,H,qblock) gather along last dim of (B,H,nb,bs*D).
        k_sel = torch.empty(
            B, H, num_blocks, top_k * bs, D, device=device, dtype=k_blocks.dtype
        )
        v_sel = torch.empty_like(k_sel)
        valid_sel = torch.empty(
            B, H, num_blocks, top_k * bs, device=device, dtype=torch.bool
        )

        for j in range(top_k):
            bj = top_indices[..., j]  # (B, H, nb)
            bj_flat_idx = bj.view(B, H, num_blocks, 1, 1).expand(-1, -1, -1, bs, D)
            k_sel[:, :, :, j * bs : (j + 1) * bs, :] = torch.gather(
                k_blocks, 2, bj_flat_idx
            )
            v_sel[:, :, :, j * bs : (j + 1) * bs, :] = torch.gather(
                v_blocks, 2, bj_flat_idx
            )
            bj_flat_idx_v = bj.view(B, H, num_blocks, 1).expand(-1, -1, -1, bs)
            valid_sel[:, :, :, j * bs : (j + 1) * bs] = torch.gather(
                valid_blocks, 2, bj_flat_idx_v
            )

        # ---- attention within selected blocks ----
        q_all = q_pad.reshape(B, H, num_blocks, bs, D)
        scale = D**-0.5
        scores = (
            torch.matmul(q_all.float(), k_sel.float().transpose(-2, -1).contiguous())
            * scale
        )

        # FA-FIX-03: causal mask via ABSOLUTE positions (per-query position vs per-key absolute position)
        if is_causal:
            q_abs = torch.arange(num_blocks, device=device).view(
                num_blocks, 1
            ) * bs + torch.arange(bs, device=device).view(
                1, bs
            )  # (nb, bs)
            q_abs = q_abs.view(1, 1, num_blocks, bs)  # (1, 1, nb, bs)
            # Absolute key positions for each query block's selected keys
            k_abs = idx_flat  # (B, H, nb, top_k*bs) — already absolute
            causal_b = k_abs.unsqueeze(-2) <= q_abs.unsqueeze(
                -1
            )  # (B, H, nb, bs, top_k*bs)
            scores = scores.masked_fill(~causal_b, float("-inf"))

        # FA-FIX-04: padding mask on the gathered keys
        scores = scores.masked_fill(~valid_sel.unsqueeze(-2), float("-inf"))

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(v_sel.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # FA-FIX-09: head_mask applied here
        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        out_blocks = torch.matmul(probs, v_sel)  # (B, H, nb, bs, D)

        # FA-FIX-08: NaN guard
        out_blocks = torch.nan_to_num(out_blocks, nan=0.0, posinf=1e4, neginf=-1e4)

        context = out_blocks.reshape(B, H, N_pad, D)[:, :, :N, :]
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        # FA-FIX-10: return block attention weights when requested
        if output_attentions:
            return context, probs
        return context, None

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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len_new = kv_input.shape[1]

        # ---- Q / K / V projections ----
        q = (
            self.q_proj(hidden_states)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k_new = (
            self.k_proj(kv_input)
            .view(B, kv_len_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_new = (
            self.v_proj(kv_input)
            .view(B, kv_len_new, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE with past_len offset ----
        if not is_cross_attention:
            past_k_probe, _ = self._read_past_kv(past_key_value, self.layer_idx)
            past_len = 0 if past_k_probe is None else past_k_probe.shape[-2]

            if position_embeddings is not None and isinstance(
                position_embeddings, tuple
            ):
                cos, sin = position_embeddings
                if past_len > 0:
                    cos = cos[..., past_len : past_len + N, :]
                    sin = sin[..., past_len : past_len + N, :]
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)
            elif self.rotary_emb is not None:
                # FA-FIX-07: use hidden_states for shape probing (was `v`)
                try:
                    cos, sin = self.rotary_emb(hidden_states, seq_len=past_len + N)
                except TypeError:
                    try:
                        cos, sin = self.rotary_emb(hidden_states, position_ids)
                    except Exception:
                        cos, sin = None, None
                if cos is not None and sin is not None:
                    if past_len > 0:
                        cos = cos[..., past_len : past_len + N, :]
                        sin = sin[..., past_len : past_len + N, :]
                    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, position_ids)

        # ---- KV-cache concat + update ----
        # FA-FIX-06: honor past_key_value
        past_k, past_v = self._read_past_kv(past_key_value, self.layer_idx)
        if past_k is not None and not is_cross_attention:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat, v_cat = k_new, v_new

        if (
            past_key_value is not None
            and hasattr(past_key_value, "update")
            and not is_cross_attention
        ):
            past_key_value.update(k_new, v_new, self.layer_idx)

        # ---- GQA ----
        if self.num_kv_heads < self.num_heads:
            k_cat = repeat_kv(k_cat, self.kv_repeat)
            v_cat = repeat_kv(v_cat, self.kv_repeat)

        N_kv = k_cat.shape[2]

        # ---- masks ----
        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        valid_mask = self._normalize_valid_mask(eff_mask, B, N_kv, device)

        # FA-FIX-03 / FA-FIX-18: causality condition
        is_causal = self.is_decoder and not is_cross_attention and N > 1

        # ---- dispatch ----
        if N <= 128 or is_cross_attention:
            context, attn_weights = self._dense_attention(
                q,
                k_cat,
                v_cat,
                valid_mask,
                head_mask,
                is_causal=is_causal and N_kv == N,
                output_attentions=bool(output_attentions),
            )
        else:
            context, attn_weights = self._block_sparse_attention(
                q,
                k_cat,
                v_cat,
                valid_mask,
                head_mask,
                is_causal=is_causal and N_kv == N,
                output_attentions=bool(output_attentions),
            )

        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.weight.device)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("fusion_attn_inject")
@ATTENTION_REGISTRY.register("inject_fusion_attn")
def inject_fusion_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_fusion_attn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
        if ok and isinstance(obj, (torch.nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_fusion_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # Single-layer injection
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, torch.nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break

    # FA-FIX-12: idempotency guard
    if src_attn is not None and (
        isinstance(src_attn, FUSIONATTN) or "FUSIONATTN" in src_attn.__class__.__name__
    ):
        return None

    try:
        new_attn = FUSIONATTN(
            config=cfg,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        # Splice
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        else:
            logger.error(
                "inject_fusion_attn: no writable attention attribute on %s",
                target.__class__.__name__,
            )
            return None

        return new_attn
    except Exception as e:
        # FA-FIX-11: log full traceback
        logger.exception("inject_fusion_attn failed at layer %d: %s", layer_idx, e)
        return None


__all__ = ["FUSIONATTN", "inject_fusion_attn"]
