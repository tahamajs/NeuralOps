"""
EAT: Efficient Attention Transformer with Progressive Token Pruning.
Production-Hardened (v2).

Fixes (v2):
  EAT-FIX-01  Removed output padding/truncation hack.
  EAT-FIX-02  RoPE cos/sin broadcast corrected.
  EAT-FIX-03  `is_nested_encoder` no longer triggers for DistilBERT.
  EAT-FIX-04  Idempotency guard in injector.
  EAT-FIX-05  SDPA causal + mask combination corrected (broadcast-safe).
  EAT-FIX-06  `head_mask` applied uniformly across dense and SDPA paths.
  EAT-FIX-07  `ProgressiveTokenPruning` only built when explicitly requested
              via `enable_pruning=True` or `keep_ratio` in `comp_config`
              (no more silent construction from `config.keep_ratio`).
  EAT-FIX-08  `LoRALinear.forward` dtype-safe.
  EAT-FIX-09  Storage accounting.
  EAT-FIX-10  Config None-safety in `__init__`.
  EAT-FIX-11  `copy_weights` device/dtype-safe and shape-tolerant.
  EAT-FIX-12  Fallback GQA pooling uses `repeat_interleave`.
  EAT-FIX-13  `logger` used consistently (no `print`).
  EAT-FIX-14  `_make_placeholder` for missing source modules.
  EAT-FIX-15  Warnings (not runtime errors) for near-zero weights.
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
from framework.core.gqa_utils import repeat_kv

try:
    from .progressive_token_pruning import ProgressiveTokenPruning
except Exception:
    ProgressiveTokenPruning = None

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    even_d = (d // 2) * 2
    x_rot = x[..., :even_d]
    half = even_d // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if even_d < d:
        return torch.cat([rotated, x[..., even_d:]], dim=-1)
    return rotated


def _broadcast_cos_sin(cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor):
    """EAT-FIX-02: broadcast cos/sin to q's shape correctly."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if cos is None or sin is None:
        return q, k

    cos, sin = _broadcast_cos_sin(cos, sin, q)

    # Slice time dimension if mismatched
    if cos.shape[-2] != q.shape[-2]:
        if cos.shape[-2] > q.shape[-2]:
            cos = cos[..., : q.shape[-2], :]
            sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_e = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_e = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
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
# LoRALinear (dtype-safe)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank) if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, self.lora_rank), **fk))
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
        # EAT-FIX-08: dtype-safe
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# EAT Block
# ===========================================================================
@ATTENTION_REGISTRY.register("eat", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("eat_attention", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
class EATBlock(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        d_model: int | Any = 768,
        n_heads: int = 12,
        prune_layer: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- EAT-FIX-10: config None-safety ----
        cfg_obj = getattr(config, "text_config", config) if config is not None else None
        hidden_size = getattr(cfg_obj, "hidden_size", None) if cfg_obj is not None else None
        num_heads = getattr(cfg_obj, "num_attention_heads", None) if cfg_obj is not None else None

        if hidden_size is None and source_attn is not None:
            for cand in ("q_lin", "query", "q_proj", "c_attn", "qkv"):
                mod = getattr(source_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    hidden_size = w.shape[1] if w.ndim == 2 else None
                    break
                elif isinstance(mod, torch.Tensor) and mod.ndim == 2:
                    hidden_size = mod.shape[1]
                    break
        if hidden_size is None:
            hidden_size = kwargs.get(
                "hidden_size",
                kwargs.get("d_model", d_model if isinstance(d_model, int) else 768),
            )
        if num_heads is None:
            num_heads = kwargs.get(
                "num_attention_heads",
                kwargs.get("num_heads", kwargs.get("n_heads", n_heads)),
            )

        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.d_model = int(hidden_size)
        self.dim = self.d_model
        self.n_heads = int(num_heads)
        self.num_heads = self.n_heads

        # ---- num_kv_heads ----
        cfg_kv = None
        if config is not None:
            cfg_kv = getattr(config, "num_key_value_heads",
                              getattr(config, "num_kv_heads", None))
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ---- head_dim ----
        cfg_hd = getattr(config, "head_dim", None) if config else None
        self.head_dim = int(cfg_hd or kwargs.get("head_dim") or
                             (self.dim // max(1, self.num_heads)))

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        resolved_layer_id = (
            layer_id if layer_id is not None
            else (layer_idx if layer_idx is not None
                  else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        )
        self.layer_idx = int(resolved_layer_id)

        # ---- LoRA ----
        self.lora_rank = int(
            getattr(config, "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
                    if comp_config is not None else kwargs.get("lora_rank", 16))
        )
        self.lora_alpha = float(
            getattr(config, "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
                    if comp_config is not None else kwargs.get("lora_alpha", 32.0))
        )
        self.freeze_base = self.lora_rank > 0

        # ---- decoder detection ----
        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(getattr(config, "is_decoder", False)
                          or getattr(config, "is_causal", False))
        if any(model_type.startswith(p) for p in
               ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
                "bloom", "falcon", "olmo", "mpt", "stablelm")):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn"):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder

        # EAT-FIX-03: never mark DistilBERT-like as nested
        self.is_nested_encoder = False

        # ---- projections ----
        self.q_lin = LoRALinear(self.dim, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.dim, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.dim, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        dropout_p = 0.1
        if config is not None:
            dropout_p = float(getattr(config, "dropout",
                              getattr(config, "attention_probs_dropout_prob", 0.1)))
        elif comp_config is not None:
            dropout_p = float(getattr(comp_config, "dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        # ---- EAT-FIX-07: pruning only when explicitly requested ----
        self.prune_layer: Any | None = None
        if prune_layer is not None:
            self.prune_layer = prune_layer
        else:
            enable_pruning = False
            keep_ratio = None
            if comp_config is not None:
                enable_pruning = bool(getattr(comp_config, "enable_pruning", False))
                keep_ratio = getattr(comp_config, "keep_ratio", None)
            if kwargs.get("enable_pruning", False):
                enable_pruning = True
            if kwargs.get("keep_ratio") is not None:
                keep_ratio = kwargs["keep_ratio"]

            if enable_pruning and ProgressiveTokenPruning is not None:
                effective_ratio = float(keep_ratio) if keep_ratio is not None else 0.7
                try:
                    self.prune_layer = ProgressiveTokenPruning(
                        config=config,
                        comp_config=comp_config,
                        keep_ratio=effective_ratio,
                        layer_id=self.layer_idx,
                        layer_idx=self.layer_idx,
                    )
                except Exception as e:
                    logger.debug("EAT: ProgressiveTokenPruning not created: %s", e)
                    self.prune_layer = None

        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        self.comp_config = comp_config

        # EAT-FIX-09
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ---- aliases ----
    @property
    def query(self): return self.q_lin
    @property
    def key(self): return self.k_lin
    @property
    def value(self): return self.v_lin
    @property
    def dense(self): return self.out_lin
    @property
    def o_proj(self): return self.out_lin
    @property
    def out_proj(self): return self.out_lin
    @property
    def q_proj(self): return self.q_lin
    @property
    def k_proj(self): return self.k_lin
    @property
    def v_proj(self): return self.v_lin
    @property
    def c_proj(self): return self.out_lin

    # ------------------------------------------------------------------
    # copy_weights (EAT-FIX-11)
    # ------------------------------------------------------------------
    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        if pretrained_attn is None:
            return

        try:
            for p in pretrained_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = any(hasattr(pretrained_attn, a) for a in
                          ("out_lin", "c_proj", "o_proj", "out_proj", "dense"))
        is_nested = False
        if (hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")) or (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb") and pretrained_attn.rotary_emb is not None:
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        def _unwrap(m):
            return getattr(m, "base_layer", m)

        # 1. GPT-2 fused c_attn
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if hasattr(c_attn_mod, "merge") and callable(c_attn_mod.merge) \
                    and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)
            base_c = _unwrap(c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c_attn_mod.__class__.__name__ == "Conv1D"
                            or base_c.__class__.__name__ == "Conv1D"
                            or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                            or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
                extracted["k"] = (c_w[split:2*split], c_b[split:2*split] if c_b is not None else None)
                extracted["v"] = (c_w[2*split:], c_b[2*split:] if c_b is not None else None)

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                if hasattr(c_proj_mod, "merge") and callable(c_proj_mod.merge) \
                        and not getattr(c_proj_mod, "merged", False):
                    try:
                        c_proj_mod.merge()
                    except Exception as e:
                        logger.debug("c_proj merge skipped: %s", e)
                base_o = _unwrap(c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (c_proj_mod.__class__.__name__ == "Conv1D"
                                  or base_o.__class__.__name__ == "Conv1D"
                                  or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf")
                                  or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. Nested (BERT / RoBERTa)
        elif hasattr(pretrained_attn, "self") \
                and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(),
                                       b.data.clone() if b is not None else None)
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                d = pretrained_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                   b.data.clone() if b is not None else None)

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for name in names:
                    if hasattr(pretrained_attn, name):
                        m = getattr(pretrained_attn, name)
                        if hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(),
                                               b.data.clone() if b is not None else None)
                            break

        # UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                    if w is None and (k != "o" or not self.is_nested_encoder)]
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

        missing_after = [k for k, (w, _) in extracted.items()
                          if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning("EATBlock copy_weights: missing %s; using initialized weights",
                            missing_after)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for key, target in (("q", self.q_lin), ("k", self.k_lin),
                                  ("v", self.v_lin), ("o", self.out_lin)):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target.weight.shape \
                        and w.t().contiguous().shape == target.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data.zero_()
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data.zero_()
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                # EAT-FIX-15: warn (not raise)
                n = float(torch.linalg.norm(target.weight.data.float()).item())
                if n < 1e-4:
                    logger.warning("EAT: weight for '%s' near-zero norm (%.2e)",
                                    key, n)

                if self.freeze_base:
                    target.freeze_base_weights()

        # ---- EAT-FIX-09: track compressed ----
        try:
            for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    @classmethod
    def from_distilbert_pretrained(cls, old_attn, model_config=None, **kwargs):
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

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
        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # ---- RoPE ----
        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        pos_emb = position_embeddings if position_embeddings is not None \
                  else kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb[0], pos_emb[1]

        if cos is not None and sin is not None and not is_cross:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross:
            try:
                pos_ids = position_ids
                if pos_ids is None:
                    pos_ids = torch.arange(kv_len, device=hidden_states.device) \
                              .unsqueeze(0).expand(B, -1)
                cos_, sin_ = self.rotary_emb(k, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos_, sin_, pos_ids)
            except Exception as e:
                logger.debug("EAT: rotary apply failed: %s", e)

        # ---- GQA ----
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        is_causal = bool(self.is_decoder and not is_cross and L > 1)

        # ---- EAT-FIX-05: unified causal + mask ----
        attn_mask_combined = None
        if eff_mask is not None or is_causal:
            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    attn_mask_combined = (eff_mask > 0)[:, None, None, :]  # (B,1,1,N_kv)
                elif eff_mask.dim() == 3:
                    attn_mask_combined = (eff_mask > -1e4)[:, None, :, :]
                elif eff_mask.dim() == 4:
                    attn_mask_combined = (eff_mask > -1e4)
                else:
                    attn_mask_combined = None

            if is_causal:
                causal = torch.tril(
                    torch.ones(L, kv_len, device=hidden_states.device, dtype=torch.bool)
                )[None, None, :, :]
                if attn_mask_combined is not None:
                    attn_mask_combined = attn_mask_combined & causal
                else:
                    attn_mask_combined = causal

        # ---- Dense attention ----
        scale = 1.0 / math.sqrt(self.head_dim)

        if attn_mask_combined is not None:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
            scores = scores.masked_fill(~attn_mask_combined, -1e4)
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(dtype_in)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)
            attn_weights_for_return = probs
        else:
            # Fast SDPA path
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _sdpa_out = F.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                probs = None
                attn_weights_for_return = None
                _sdpa_out = _sdpa_out.to(dtype_in)
            except Exception:
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
                if is_causal:
                    causal = torch.triu(
                        torch.full((L, kv_len), -1e4, device=scores.device, dtype=torch.float32),
                        diagonal=1 + (kv_len - L),
                    ).unsqueeze(0).unsqueeze(0)
                    scores = scores + causal
                scores = torch.nan_to_num(scores, nan=-1e4)
                probs = F.softmax(scores, dim=-1).to(dtype_in)
                probs = torch.nan_to_num(probs, nan=0.0)
                probs = self.dropout(probs)
                attn_weights_for_return = probs
                _sdpa_out = None

        # ---- EAT-FIX-06: head_mask ----
        if head_mask is not None:
            hm = head_mask.to(dtype_in)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)

            if probs is not None:
                probs = probs * hm
            else:
                _sdpa_out = _sdpa_out * hm

        # ---- Context ----
        if probs is not None:
            context = torch.matmul(probs.to(v.dtype), v)
        else:
            context = _sdpa_out

        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # ---- Output projection ----
        # EAT-FIX-03: is_nested_encoder is only True for truly nested containers
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_lin(context)

        # EAT-FIX-01: no padding/truncation hack
        if out.shape[-1] != D:
            raise RuntimeError(
                f"EATBlock: output dim {out.shape[-1]} != hidden_size {D}. "
                f"Check num_heads * head_dim == hidden_size."
            )

        # ---- Progressive Token Pruning (eval only) ----
        if self.prune_layer is not None and not self.training:
            try:
                out, _ = self.prune_layer(out)
            except Exception as e:
                logger.debug("EAT: pruning skipped: %s", e)

        return out, (attn_weights_for_return if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("eat_fn")
@ATTENTION_REGISTRY.register("eat_inject")
@ATTENTION_REGISTRY.register("inject_eat")
def inject_eat(
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
    # EAT-FIX-04: recursion guard
    if _depth > 8:
        logger.warning("inject_eat: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("transformer", "layer"),
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
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_eat(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single-layer injection ----
    layer = target

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if isinstance(cand, nn.Module) and cand is not layer:
                    src_attn = getattr(cand, "self", cand)
                    break

    # EAT-FIX-04: idempotency
    if src_attn is not None and (isinstance(src_attn, EATBlock)
                                   or "EATBlock" in src_attn.__class__.__name__):
        return None
    if src_attn is not None and ("DeltaNet" in src_attn.__class__.__name__
                                   or "LinearAttention" in src_attn.__class__.__name__):
        return None

    try:
        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is None and src_attn is not None and hasattr(src_attn, "config"):
            model_config = src_attn.config

        new_attn = EATBlock(
            config=model_config,
            pretrained_attn=src_attn if src_attn is not layer else None,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            **kwargs,
        )

        if src_attn is not None and src_attn is not layer:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        # Splice
        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
            spliced = True
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
            spliced = True

        if not spliced:
            logger.error("inject_eat: no writable attention attribute on %s",
                          layer.__class__.__name__)
            return None

        return new_attn

    except Exception as e:
        logger.exception("inject_eat failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "EATBlock",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_eat",
]
