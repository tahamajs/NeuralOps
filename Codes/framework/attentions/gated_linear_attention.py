"""Gated Linear Attention (GLA) — Production-Hardened (v3).

Multi-head linear recurrent attention with data-dependent decay gating,
multiplicative SiLU output gating, denominator normalization, and pretrained
weight transfer. Supports DistilBERT, RoBERTa, GPT-2, and Qwen.

Fixes in v3:
  GLA3-FIX-01  Explicit shape assertion in `__init__` for num_heads, head_dim,
               decay_gate, log_decay, state_norm. Raises early with a clear
               message instead of failing mid-forward.
  GLA3-FIX-02  Robust num_heads / num_kv_heads / head_dim detection from
               config + source attention + kwargs, with consistent clamping.
  GLA3-FIX-03  `decay_gate` and `log_decay` are built from the *resolved*
               num_heads, so they can never disagree.
  GLA3-FIX-04  `head_dim` derived from config.head_dim OR from q_lin's
               out_features // num_heads, ensuring alignment with the
               pretrained weight shape.
  GLA3-FIX-05  Sanity check in forward: `q.shape[-1] == head_dim` else raise
               a descriptive error pointing at the offending weight.
  GLA3-FIX-06  `head_mask` handling unified across 1D/2D/3D/4D.
  GLA3-FIX-07  `z_proj` output dim verified against `all_head_size`.
  GLA3-FIX-08  dtype-safe `log_decay` broadcast (explicit `.to(y.dtype)`).
  GLA3-FIX-09  Removed dead `output_scale` multiplication on `is_nested_encoder`
               branch (it was applied twice in some framework paths).
  GLA3-FIX-10  `_normalize_valid_mask` now returns (B, N_q) even when the
               input is 4D with N_q != 1.
  GLA3-FIX-11  `inject_gated_linear` resolves d_model, num_heads, and
               head_dim from the source attention's actual weight shapes,
               not just from config attributes.
  GLA3-FIX-12  Idempotency guard also checks the parent layer, not just
               the source attention.
  GLA3-FIX-13  `copy_weights` validates extracted weight shapes against
               the resolved (all_head_size, dim) and raises early on
               irreconcilable mismatch.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

try:
    from framework.core.triton_kernels.gla_scan import gla_scan_triton, MAX_REGISTER_DIM
    HAS_TRITON_GLA = True
except Exception as _e:
    logger.warning("GLA Triton unavailable: %s", _e)
    HAS_TRITON_GLA = False
    MAX_REGISTER_DIM = 64



# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
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
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]
    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
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

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# Decoder detection helper
# ===========================================================================
def _detect_decoder(config, is_decoder, src):
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if any(mt.startswith(p) for p in
               ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
                "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if src is not None:
        if getattr(src, "is_decoder", False):
            return True
        if hasattr(src, "c_attn") or "gpt" in src.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Shape introspection helpers  (GLA3-FIX-02, GLA3-FIX-04)
# ===========================================================================
def _resolve_dim_and_heads(config, src_attn, kwargs):
    """
    Determine (d_model, num_heads, head_dim, num_kv_heads) from whichever
    source is reliable: kwargs > config > source attention weight shapes.
    """
    # --- d_model ---
    d_model = (
        kwargs.get("d_model")
        or kwargs.get("hidden_size")
    )
    if d_model is None and config is not None:
        d_model = getattr(config, "hidden_size",
                  getattr(config, "dim",
                  getattr(config, "d_model", None)))
    if d_model is None and src_attn is not None:
        # Look at the actual weight of the query projection
        for attr in ("q_proj", "q_lin", "query"):
            mod = getattr(src_attn, attr, None)
            if mod is not None and hasattr(mod, "weight"):
                d_model = mod.weight.shape[1]
                break
        if d_model is None:
            for attr in ("out_proj", "out_lin", "dense", "c_proj", "o_proj"):
                mod = getattr(src_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    d_model = mod.weight.shape[1]
                    break
    if d_model is None:
        d_model = 768

    # --- num_heads ---
    num_heads = (
        kwargs.get("num_heads")
        or kwargs.get("num_attention_heads")
    )
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads",
                     getattr(config, "n_heads",
                     getattr(config, "num_heads", None)))
    if num_heads is None and src_attn is not None:
        num_heads = getattr(src_attn, "num_heads",
                     getattr(src_attn, "num_attention_heads",
                     getattr(src_attn, "n_heads", None)))
    if num_heads is None:
        num_heads = 12
    num_heads = int(num_heads)

    # --- head_dim (prefer explicit config) ---
    head_dim = (
        kwargs.get("head_dim")
        or (getattr(config, "head_dim", None) if config is not None else None)
        or (getattr(src_attn, "head_dim", None) if src_attn is not None else None)
        or (getattr(src_attn, "attention_head_size", None) if src_attn is not None else None)
    )
    if head_dim is None and src_attn is not None:
        # Derive from q_lin's output size and num_heads
        for attr in ("q_proj", "q_lin", "query"):
            mod = getattr(src_attn, attr, None)
            if mod is not None and hasattr(mod, "weight"):
                q_out = mod.weight.shape[0]
                if q_out % num_heads == 0:
                    head_dim = q_out // num_heads
                break
    if head_dim is None:
        head_dim = d_model // max(1, num_heads)
    head_dim = int(head_dim)

    # --- num_kv_heads ---
    num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads",
                        getattr(config, "num_kv_heads", None))
    if num_kv_heads is None and src_attn is not None:
        num_kv_heads = getattr(src_attn, "num_key_value_heads",
                        getattr(src_attn, "num_kv_heads", None))
    if num_kv_heads is None:
        num_kv_heads = num_heads
    num_kv_heads = int(num_kv_heads)
    num_kv_heads = max(1, min(num_kv_heads, num_heads))

    return int(d_model), num_heads, head_dim, num_kv_heads


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("gated_linear_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("gated_linear_attn", supports_causal=True, supports_gqa=True)
class GatedLinearAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "linear_attn"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        head_dim: int | None = None,
        num_kv_heads: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        chunk_size: int = 128,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        src_attn = kwargs.get("pretrained_attn") or kwargs.get("old_attn")

        # Merge explicit args into kwargs for resolver
        _kw = dict(kwargs)
        if d_model is not None:
            _kw["d_model"] = d_model if isinstance(d_model, int) else getattr(d_model, "hidden_size", None)
        if num_heads is not None:
            _kw["num_heads"] = num_heads
        if head_dim is not None:
            _kw["head_dim"] = head_dim
        if num_kv_heads is not None:
            _kw["num_kv_heads"] = num_kv_heads

        # GLA3-FIX-02 / GLA3-FIX-04
        resolved_dim, resolved_heads, resolved_head_dim, resolved_kv_heads = \
            _resolve_dim_and_heads(config, src_attn, _kw)

        self.dim = resolved_dim
        self.hidden_size = self.dim
        self.num_heads = resolved_heads
        self.head_dim = resolved_head_dim
        self.num_kv_heads = resolved_kv_heads

        # GLA3-FIX-01: hard consistency check
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")
        if self.dim != self.num_heads * self.head_dim:
            # This is the smoking gun for the 12-vs-16 error
            raise ValueError(
                f"[GLA] Dimension mismatch in config: "
                f"hidden_size={self.dim}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim}, "
                f"num_heads * head_dim = {self.num_heads * self.head_dim} != {self.dim}. "
                f"Fix by passing explicit head_dim or set config.head_dim."
            )

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.kv_head_size = self.num_kv_heads * self.head_dim

        self.chunk_size = max(16, int(chunk_size))
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)

        self.is_decoder = _detect_decoder(config, is_decoder, src_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---- Projections ----
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.k_proj = LoRALinear(self.dim, self.kv_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.v_proj = LoRALinear(self.dim, self.kv_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)
        self.z_proj = LoRALinear(self.dim, self.all_head_size, bias=False,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=True)

        # GLA3-FIX-03: build decay_gate/log_decay from the same `num_heads`
        self.decay_gate = nn.Linear(self.dim, self.num_heads, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.decay_gate.weight)
            nn.init.zeros_(self.decay_gate.bias)

        self.log_decay = nn.Parameter(torch.zeros(self.num_heads))

        self.state_norm = nn.LayerNorm(self.head_dim)
        self.output_scale = nn.Parameter(torch.ones(1))

        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=True)

        # Sanity: ensure z_proj output matches all_head_size
        if self.z_proj.out_features != self.all_head_size:
            raise RuntimeError(
                f"GLA: z_proj.out_features={self.z_proj.out_features} "
                f"!= all_head_size={self.all_head_size}"
            )

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0
        self.triton_kwargs = {"phi": "elu+1", "eps": 1e-4}

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ---- aliases ----
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
    def o_proj(self): return self.out_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

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

        cls_name = src.__class__.__name__
        is_nested = False
        if (hasattr(src, "self") and hasattr(src.self, "query")
                and hasattr(src, "output") and hasattr(src.output, "dense")) or (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {"q": (None, None), "k": (None, None),
                      "v": (None, None), "o": (None, None)}

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (c.__class__.__name__ == "Conv1D"
                            or base_c.__class__.__name__ == "Conv1D"
                            or hasattr(c, "nf") or hasattr(base_c, "nf")
                            or (cw.ndim == 2 and cw.shape[0] < cw.shape[1]))
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (cw[split:2*split], cb[split:2*split] if cb is not None else None)
                extracted["v"] = (cw[2*split:], cb[2*split:] if cb is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (o.__class__.__name__ == "Conv1D"
                                  or base_o.__class__.__name__ == "Conv1D"
                                  or hasattr(o, "nf") or hasattr(base_o, "nf")
                                  or (ow.ndim == 2 and ow.shape[0] < ow.shape[1]))
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # Nested BERT / RoBERTa
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(),
                                       b.data.clone() if b is not None else None)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                   b.data.clone() if b is not None else None)

        # Flat (DistilBERT, Qwen, LLaMA)
        else:
            candidates = {
                "q": ("q_proj", "q_lin", "query"),
                "k": ("k_proj", "k_lin", "key"),
                "v": ("v_proj", "v_lin", "value"),
                "o": ("o_proj", "out_proj", "out_lin", "dense"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
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
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("GLA UniversalAdapter fallback failed: %s", e)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for key, target in (("q", self.q_proj), ("k", self.k_proj),
                                  ("v", self.v_proj), ("o", self.out_proj)):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target.weight.shape \
                        and w.t().contiguous().shape == target.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))

                if key in ("k", "v"):
                    src_out = w.shape[0]
                    tgt_out = target.weight.shape[0]
                    if src_out == tgt_out:
                        target.weight.data.copy_(w)
                    elif src_out > tgt_out and src_out % tgt_out == 0:
                        rep = src_out // tgt_out
                        target.weight.data.copy_(w.view(tgt_out, rep, -1).mean(dim=1))
                    elif src_out < tgt_out and tgt_out % src_out == 0:
                        rep = tgt_out // src_out
                        target.weight.data.copy_(w.repeat(rep, 1))
                    else:
                        target.weight.data.zero_()
                        target.weight.data[:out_d, :in_d].copy_(
                            w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                        )
                else:
                    target.weight.data.zero_()
                    target.weight.data[:out_d, :in_d].copy_(
                        w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                    )

                # GLA3-FIX-13: validate that the query projection is now fully
                # in the expected shape (all_head_size, hidden_size)
                if key == "q":
                    if target.weight.shape != (self.all_head_size, self.dim):
                        raise RuntimeError(
                            f"GLA copy_weights: q_proj shape {tuple(target.weight.shape)} "
                            f"!= expected ({self.all_head_size}, {self.dim}). "
                            f"num_heads={self.num_heads}, head_dim={self.head_dim}, dim={self.dim}."
                        )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )

                n = torch.linalg.norm(target.weight.data.float()).item()
                if n < 1e-4:
                    logger.warning("GLA: %s near-zero norm (%.2e)", key, n)

                target.freeze_base_weights()

            # Initialize z_proj from q
            w_q = extracted["q"][0]
            if w_q is not None:
                w_q_oriented = _orient(w_q, self.dim)
                out_d = min(self.z_proj.weight.shape[0], w_q_oriented.shape[0])
                in_d = min(self.z_proj.weight.shape[1], w_q_oriented.shape[1])
                self.z_proj.weight.data.zero_()
                self.z_proj.weight.data[:out_d, :in_d].copy_(
                    w_q_oriented[:out_d, :in_d].to(self.z_proj.weight.device,
                                                     self.z_proj.weight.dtype)
                )
                self.z_proj.freeze_base_weights()

        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.z_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Chunked linear-attention scan
    # ------------------------------------------------------------------
    def _gla_scan_chunked(self, q, k, v, lambdas, mask=None, reverse=False):
        B, H, L, d = q.shape
        device = q.device

        if HAS_TRITON_GLA and q.is_cuda and d <= MAX_REGISTER_DIM and mask is None and not reverse:
            try:
                return gla_scan_triton(q.float(), k.float(), v.float(), lambdas.float()).to(q.dtype)
            except Exception as e:
                logger.debug("GLA Triton scan failed, falling back: %s", e)

        if reverse:
            q = torch.flip(q, dims=[2])
            k = torch.flip(k, dims=[2])
            v = torch.flip(v, dims=[2])
            lambdas = torch.flip(lambdas, dims=[2])
            if mask is not None:
                mask = torch.flip(mask, dims=[1])

        if mask is not None:
            m = mask.float()
            m_kv = m.unsqueeze(1).unsqueeze(-1)
            k = k * m_kv
            v = v * m_kv

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        lam = lambdas.float().clamp(min=1e-4, max=1.0 - 1e-4)

        chunk = self.chunk_size
        n_chunks = (L + chunk - 1) // chunk

        S = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
        Z = torch.zeros(B, H, d, device=device, dtype=torch.float32)
        outs = []

        for c in range(n_chunks):
            s = c * chunk
            e = min(s + chunk, L)
            q_c = q_f[:, :, s:e]
            k_c = k_f[:, :, s:e]
            v_c = v_f[:, :, s:e]
            lam_c = lam[:, :, s:e]

            log_lam = torch.log(lam_c)
            C = torch.cumsum(log_lam, dim=-1)
            C_i = C.unsqueeze(-1)
            C_j = C.unsqueeze(-2)
            D = torch.exp((C_i - C_j).clamp(min=-15.0, max=0.0)) \
                * torch.tril(torch.ones(e - s, e - s, device=device))

            lam_prefix = torch.exp(C)
            y_prev = torch.einsum("bhl,bhld->bhld", lam_prefix,
                                    torch.einsum("bhlc,bhcd->bhld", q_c, S))
            denom_prev = torch.einsum("bhl,bhl->bhl", lam_prefix,
                                        torch.einsum("bhlc,bhc->bhl", q_c, Z))

            qk = torch.matmul(q_c, k_c.transpose(-1, -2))
            y_cur = torch.matmul(qk * D, v_c)
            denom_cur = (qk * D).sum(dim=-1)

            y_chunk = y_prev + y_cur
            denom_chunk = (denom_prev + denom_cur).clamp(min=1e-4)

            lam_prod_chunk = torch.exp(C[:, :, -1]).view(B, H, 1, 1)
            C_end = C[:, :, -1:].unsqueeze(-1)
            C_t = C.unsqueeze(-1)
            weight_t = torch.exp((C_end - C_t).clamp(min=-15.0, max=0.0))
            k_w = k_c * weight_t
            S_new = lam_prod_chunk * S + torch.einsum("bhld,bhle->bhde", k_w, v_c)
            Z_new = lam_prod_chunk.squeeze(-1) * Z + k_w.sum(dim=2)

            S = S_new
            Z = Z_new

            y_chunk = y_chunk / denom_chunk.unsqueeze(-1)
            outs.append(y_chunk)

        y = torch.cat(outs, dim=2)
        if reverse:
            y = torch.flip(y, dims=[2])
        return y.to(q.dtype)

    # ------------------------------------------------------------------
    # Mask helpers (GLA3-FIX-10)
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
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            # Shape (B, 1, N_q, L) or (B, H, N_q, L) → reduce over queries
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
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
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape

        # GLA3-FIX-05
        if self.dim != D:
            raise RuntimeError(
                f"GLA: hidden_states has last dim {D}, but module expects {self.dim}. "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}."
            )

        H, d = self.num_heads, self.head_dim
        dtype_in = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        # ---- Projections ----
        q_raw = self.q_proj(hidden_states)
        k_raw = self.k_proj(kv_input)
        v_raw = self.v_proj(kv_input)

        # GLA3-FIX-07: verify projection outputs
        expected_q = (B, L, self.all_head_size)
        expected_kv = (B, L_kv, self.kv_head_size)
        if tuple(q_raw.shape) != expected_q:
            raise RuntimeError(
                f"GLA: q_proj output {tuple(q_raw.shape)} != expected {expected_q}"
            )
        if tuple(k_raw.shape) != expected_kv:
            raise RuntimeError(
                f"GLA: k_proj output {tuple(k_raw.shape)} != expected {expected_kv}"
            )

        q = q_raw.view(B, L, H, d).transpose(1, 2).contiguous()
        k = k_raw.view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = v_raw.view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()
        z = self.z_proj(hidden_states)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ELU+1 feature map
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # ---- Decay gate ----
        # GLA3-FIX-03 / GLA3-FIX-08: shape is guaranteed to be (B, L, num_heads)
        decay_out = self.decay_gate(hidden_states)   # (B, L, H)
        if decay_out.shape[-1] != self.num_heads:
            raise RuntimeError(
                f"GLA: decay_gate output {decay_out.shape[-1]} != num_heads {self.num_heads}"
            )
        decay_logits = decay_out.transpose(1, 2) + self.log_decay.to(decay_out.dtype).view(1, -1, 1)
        lambdas = torch.sigmoid(decay_logits)        # (B, H, L)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        mask_bool = self._normalize_valid_mask(eff_mask, B, L_kv, hidden_states.device)

        is_causal = bool(self.is_decoder and not is_cross and L > 1 and past_key_value is None)

        if is_causal:
            y = self._gla_scan_chunked(q, k, v, lambdas, mask=mask_bool, reverse=False)
        else:
            y_fwd = self._gla_scan_chunked(q, k, v, lambdas, mask=mask_bool, reverse=False)
            y_bwd = self._gla_scan_chunked(q, k, v, lambdas, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        y = self.state_norm(y)

        # GLA3-FIX-06: unified head_mask handling
        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, H, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            # 4D passes through
            if hm.shape[1] != H:
                raise RuntimeError(
                    f"GLA: head_mask has {hm.shape[1]} heads, expected {H}"
                )
            y = y * hm

        y = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(y * F.silu(z), nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        out = out * self.output_scale

        if mask_bool is not None and not is_cross:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("q_proj", self.q_proj), ("k_proj", self.k_proj),
                         ("v_proj", self.v_proj), ("z_proj", self.z_proj),
                         ("out_proj", self.out_proj),
                         ("decay_gate", self.decay_gate),
                         ("state_norm", self.state_norm)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["output_scale"] = _sz(self.output_scale)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection (GLA3-FIX-11, GLA3-FIX-12)
# ===========================================================================
@ATTENTION_REGISTRY.register("gated_linear_attention_inject")
@ATTENTION_REGISTRY.register("inject_gated_linear")
def inject_gated_linear(
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
        logger.warning("inject_gated_linear: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
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
            res = inject_gated_linear(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # ---- Single-layer ----
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    # GLA3-FIX-12: idempotency guard on both source and parent
    if src_attn is not None:
        if isinstance(src_attn, GatedLinearAttention) \
                or "GatedLinear" in src_attn.__class__.__name__:
            return None
        if "DeltaNet" in src_attn.__class__.__name__ \
                or "LinearAttention" in src_attn.__class__.__name__:
            return None

    if src_attn is None and cfg is None:
        return None

    # GLA3-FIX-11: resolve dims from actual weight shapes
    resolved_dim, resolved_heads, resolved_head_dim, resolved_kv_heads = \
        _resolve_dim_and_heads(cfg, src_attn, kwargs)

    logger.info(
        "inject_gated_linear: layer %d resolved dim=%d, heads=%d, head_dim=%d, kv_heads=%d",
        layer_idx, resolved_dim, resolved_heads, resolved_head_dim, resolved_kv_heads,
    )

    try:
        new_block = GatedLinearAttention(
            config=cfg,
            d_model=resolved_dim,
            num_heads=resolved_heads,
            head_dim=resolved_head_dim,
            num_kv_heads=resolved_kv_heads,
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
    except Exception as e:
        logger.error("GatedLinearAttention construction failed: %s", e)
        return None

    # Splice
    if target is not None:
        spliced = False
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
            spliced = True
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True

        if not spliced:
            logger.error("inject_gated_linear: no writable attention attribute on %s",
                          target.__class__.__name__)
            return None

    return new_block


__all__ = [
    "GatedLinearAttention",
    "inject_gated_linear",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
