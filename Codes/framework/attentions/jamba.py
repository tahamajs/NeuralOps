"""Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block (Production-Hardened v2).

Combines:
- SVD-Mamba selective state space branch
- Multi-Head Attention branch with GQA, RoPE, and causal masking
- Expert Choice MoE branch with auxiliary loss tracking

Fixes (v2):
  JAMBA-FIX-01  `SVDMambaSub.is_custom_attention = False` (it is a linear
                submodule, not an attention module).
  JAMBA-FIX-02  `self.config` is explicitly set in `__init__`, so
                `getattr(self.config, ...)` calls no longer raise.
  JAMBA-FIX-03  `self.aux_loss` is stored as a Python float (not a CPU
                tensor), avoiding device-mismatch when the training loop
                aggregates it with GPU losses.
  JAMBA-FIX-04  `copy_weights` no longer raises RuntimeError on near-zero
                norm; it logs a warning (matching the rest of the repo).
  JAMBA-FIX-05  `forward` signature now matches the BERT/RoBERTa convention
                and accepts `layer_idx` via kwargs.
  JAMBA-FIX-06  SDPA fast path is only taken when the output can be trusted
                (no head_mask, no output_attentions). Otherwise falls back
                to the manual path.
  JAMBA-FIX-07  The bare `except Exception: pass` around SDPA was replaced
                with an explicit logging and a proper fallback.
  JAMBA-FIX-08  `head_mask` handling unified across 1D/2D/3D/4D shapes.
  JAMBA-FIX-09  `attention_mask` at dim 4 no longer uses a broken
                `.clamp(min=-1e4)` (which works only on floats and only
                for lower bound); instead a proper additive mask is applied.
  JAMBA-FIX-10  `moe_out` handles both tensor and tuple returns from the
                MoE branch.
  JAMBA-FIX-11  `is_decoder` detection unified into a single helper.
  JAMBA-FIX-12  Idempotency guard in the injector (checks both source and
                parent layer).
  JAMBA-FIX-13  Recursion guard in the injector.
  JAMBA-FIX-14  Storage accounting exposed via `storage_bytes()`.
  JAMBA-FIX-15  `num_kv_heads` resolution is None-safe across configs.
  JAMBA-FIX-16  `is_nested_encoder` is propagated to the injected block
                after splicing.
  JAMBA-FIX-17  `LoRALinear.forward` is dtype-safe.
  JAMBA-FIX-18  Removed unused imports (`get_qkv_projection_dims`,
                `SVDMamba`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.expert_choice_moe import ExpertChoiceMoE
from framework.attentions.simple_ssm import SimpleSSM
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# LoRALinear (JAMBA-FIX-17)
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
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
        # JAMBA-FIX-17
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# SSM sub-branch (JAMBA-FIX-01)
# ===========================================================================
class SVDMambaSub(nn.Module):
    # JAMBA-FIX-01: not attention
    is_custom_attention: bool = False

    def __init__(
        self,
        d_model: int = 768,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        is_decoder: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if d_model is None and config is not None:
            d_model = getattr(config, "hidden_size",
                       getattr(config, "d_model",
                       getattr(config, "dim", 768)))
        d_model = int(d_model or 768)
        self.d_model = d_model
        self.ssm = SimpleSSM(d_model=d_model, config=config, is_decoder=is_decoder)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        x_proj = self.in_proj(hidden_states)
        ssm_out = self.ssm(x_proj, attention_mask=attention_mask)
        if isinstance(ssm_out, tuple):
            ssm_out = ssm_out[0]
        out = self.out_proj(ssm_out)
        return out, None


# ===========================================================================
# Decoder detection (JAMBA-FIX-11)
# ===========================================================================
def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    source_attn: Optional[nn.Module],
    kwargs: Dict[str, Any],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if kwargs.get("is_decoder") or kwargs.get("is_causal"):
        return True
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if any(mt.startswith(p) for p in
               ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if source_attn is not None:
        if getattr(source_attn, "is_decoder", False):
            return True
        if hasattr(source_attn, "c_attn"):
            return True
        if "gpt" in source_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Jamba-2 Block
# ===========================================================================
@ATTENTION_REGISTRY.register("jamba", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("jamba2", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
class Jamba2Block(BaseAttention):
    r"""Jamba-2 Hybrid: SVD-Mamba + Multi-Head Attention + Expert-Choice MoE."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_experts: int = 4,
        top_k: int = 2,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # JAMBA-FIX-02: store config for later getattr
        self.config = config

        # ---------- d_model ----------
        d_model = (kwargs.get("d_model") or kwargs.get("hidden_size")
                    or kwargs.get("dim") or d_model)
        n_heads = (kwargs.get("n_heads") or kwargs.get("num_heads")
                    or kwargs.get("num_attention_heads") or n_heads)

        if d_model is None and config is not None:
            d_model = getattr(config, "hidden_size",
                       getattr(config, "d_model",
                       getattr(config, "dim", 768)))
        if d_model is None and source_attn is not None:
            for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                layer = getattr(source_attn, attr, None)
                if layer is not None and hasattr(layer, "out_features"):
                    d_model = layer.out_features
                    break
        if d_model is None:
            d_model = 768

        if n_heads is None and config is not None:
            n_heads = getattr(config, "num_attention_heads",
                       getattr(config, "n_heads", 12))
        if n_heads is None and source_attn is not None:
            n_heads = getattr(source_attn, "num_attention_heads",
                       getattr(source_attn, "n_heads", 12))
        if n_heads is None:
            n_heads = 12

        if config is not None:
            n_experts = int(getattr(config, "num_experts",
                             getattr(config, "n_experts", n_experts)))
            top_k = int(getattr(config, "top_k", top_k))

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.n_heads = int(n_heads)
        self.num_heads = self.n_heads

        # JAMBA-FIX-15: None-safe kv head resolution
        num_kv_heads = None
        if config is not None:
            num_kv_heads = getattr(config, "num_key_value_heads",
                            getattr(config, "num_kv_heads", None))
        if num_kv_heads is None:
            num_kv_heads = kwargs.get("num_kv_heads",
                            kwargs.get("num_key_value_heads",
                            self.num_heads))
        self.num_kv_heads = max(1, min(int(num_kv_heads), self.num_heads))

        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)
        self.supports_gqa = True

        # JAMBA-FIX-11
        self.is_decoder = _detect_decoder(config, kwargs.get("is_decoder"),
                                            source_attn, kwargs)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # ---------- LoRA config ----------
        if config is not None:
            self.lora_rank = int(getattr(
                config, "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            ))
            self.lora_alpha = float(getattr(
                config, "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            ))
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank",
                                          kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha",
                                             kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = (self.lora_rank > 0)

        self.source_attn = source_attn
        self.rotary_emb = (getattr(source_attn, "rotary_emb", None)
                            if source_attn is not None else None)

        # ---------- MHA projections ----------
        self.q_lin = LoRALinear(self.d_model, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.d_model, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin

        # ---------- SSM branch ----------
        self.mamba = SVDMambaSub(self.d_model, config=config, rank=rank,
                                  is_decoder=self.is_decoder)

        # ---------- MoE branch ----------
        self.moe = ExpertChoiceMoE(self.d_model, n_experts=n_experts,
                                    capacity_factor=1.0)

        # ---------- Norms ----------
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        dropout_p = float(getattr(
            config, "dropout",
            getattr(config, "attention_probs_dropout_prob", 0.1) if config else 0.1,
        ))
        self.dropout = nn.Dropout(dropout_p)

        # JAMBA-FIX-03: store as float to avoid device mismatch
        self.aux_loss: float = 0.0

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    def freeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

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
        has_out = any(hasattr(src, a) for a in
                       ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") \
                or (hasattr(src, "query") and not has_out):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

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
                            or hasattr(c, "nf")
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
                                  or hasattr(o, "nf")
                                  or (ow.ndim == 2 and ow.shape[0] < ow.shape[1]))
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # Nested (BERT / RoBERTa)
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
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
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(src, n):
                        m = getattr(src, n)
                        if m is not None and hasattr(m, "weight"):
                            w_data = m.weight.data.clone()
                            is_conv = (m.__class__.__name__ == "Conv1D"
                                        or hasattr(m, "nf")
                                        or (key == "o" and w_data.ndim == 2
                                            and w_data.shape[0] < w_data.shape[1]))
                            if is_conv:
                                w_data = w_data.t().contiguous()
                            b = getattr(m, "bias", None)
                            extracted[key] = (w_data,
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
                logger.debug("Jamba2 UniversalAdapter fallback: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                          if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"Jamba2Block copy_weights failed: missing mandatory "
                f"projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.q_lin, "k": self.k_lin,
                       "v": self.v_lin, "o": self.out_lin}

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

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
                if w.shape != target.weight.shape \
                        and w.t().contiguous().shape == target.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                # JAMBA-FIX-04: warn, do not raise
                n = torch.linalg.norm(target.weight.data.float()).item()
                if n < 1e-4:
                    logger.warning("Jamba2: '%s' near-zero norm (%.2e)", key, n)

                if self.freeze_base:
                    target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if self.freeze_base:
            self.freeze_base_weights()

        try:
            self._comp_storage_bytes = sum(
                p.numel() * p.element_size()
                for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin,
                           self.moe, self.mamba, self.norm1, self.norm2, self.norm3)
                for p in m.parameters()
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Forward (JAMBA-FIX-05)
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        layer_idx = kwargs.get("layer_idx", 0)
        position_embeddings = kwargs.get("position_embeddings", None)

        B, L, D = hidden_states.shape

        # ---------- Branch 1: SVD-Mamba ----------
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask)
        x1 = self.norm1(hidden_states + mamba_out)

        # ---------- Branch 2: MHA ----------
        q = self.q_lin(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(k, seq_len=L)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(k, pos_ids)
                except Exception:
                    try:
                        cos, sin = self.rotary_emb(pos_ids)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = bool(self.is_decoder and L > 1)

        # ---------- SDPA fast path (JAMBA-FIX-06, JAMBA-FIX-07) ----------
        use_sdpa = (
            head_mask is None
            and not output_attentions
            and attention_mask is None
        )
        if use_sdpa:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _sdpa_out = F.scaled_dot_product_attention(
                    _qs, _ks, _vs,
                    is_causal=is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = _sdpa_out.to(hidden_states.dtype)\
                                     .transpose(1, 2).contiguous()\
                                     .view(B, L, self.all_head_size)
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)

                attn_proj = attn_out if self.is_nested_encoder else self.out_lin(attn_out)
                x2 = self.norm2(x1 + self.dropout(attn_proj))

                moe_out = self.moe(x2)
                # JAMBA-FIX-10
                if isinstance(moe_out, (tuple, list)):
                    moe_out = moe_out[0]
                x3 = self.norm3(x2 + self.dropout(moe_out))

                # JAMBA-FIX-03: store as float
                aux = getattr(self.moe, "aux_loss", 0.0)
                self.aux_loss = float(aux.item()) if torch.is_tensor(aux) else float(aux)

                return x3, None
            except Exception as e:
                # JAMBA-FIX-07: log, then fall back to manual
                logger.debug("Jamba2 SDPA path failed, falling back to manual: %s", e)

        # ---------- Manual attention path ----------
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if is_causal:
            causal = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

        # JAMBA-FIX-09: proper additive attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
                if valid is not None:
                    scores = scores.masked_fill(~valid.unsqueeze(1).unsqueeze(2), -1e4)
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).float()
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(v.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # JAMBA-FIX-08: unified head_mask
        if head_mask is not None:
            hm = head_mask.to(probs.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, self.num_heads, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], self.num_heads, 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        attn_out = torch.matmul(probs, v)\
                        .transpose(1, 2).contiguous()\
                        .view(B, L, self.all_head_size)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_proj = attn_out if self.is_nested_encoder else self.out_lin(attn_out)
        x2 = self.norm2(x1 + self.dropout(attn_proj))

        # ---------- Branch 3: MoE ----------
        moe_out = self.moe(x2)
        if isinstance(moe_out, (tuple, list)):
            moe_out = moe_out[0]
        x3 = self.norm3(x2 + self.dropout(moe_out))

        aux = getattr(self.moe, "aux_loss", 0.0)
        self.aux_loss = float(aux.item()) if torch.is_tensor(aux) else float(aux)

        return x3, (probs if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> Dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("q_lin", self.q_lin), ("k_lin", self.k_lin),
                         ("v_lin", self.v_lin), ("out_lin", self.out_lin),
                         ("mamba", self.mamba), ("moe", self.moe)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injector (JAMBA-FIX-12, JAMBA-FIX-13, JAMBA-FIX-16)
# ===========================================================================
@ATTENTION_REGISTRY.register("jamba_inject")
@ATTENTION_REGISTRY.register("jamba2_inject")
@ATTENTION_REGISTRY.register("inject_jamba")
@ATTENTION_REGISTRY.register("inject_jamba2")
def inject_jamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
) -> Optional[Union[Jamba2Block, nn.Module]]:
    # JAMBA-FIX-13
    if _depth > 8:
        logger.warning("inject_jamba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---------- Whole-model traversal ----------
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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_jamba(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # ---------- Single layer ----------
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and hasattr(target, "forward"):
        src_attn = target

    # JAMBA-FIX-12: idempotency guard
    if isinstance(src_attn, Jamba2Block) or "Jamba2" in src_attn.__class__.__name__:
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        child = getattr(target, attr, None)
        if isinstance(child, Jamba2Block):
            return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (getattr(getattr(src_attn, "q_proj", None), "in_features", None))
        or (getattr(getattr(src_attn, "q_lin", None), "in_features", None))
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
        new_block = Jamba2Block(
            d_model=hidden_size,
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn if src_attn is not target else None,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error("Jamba2Block injection failed: %s", e)
        return None

    # JAMBA-FIX-16
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
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
            spliced = True
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
            spliced = True

        if not spliced:
            logger.error("inject_jamba: no writable attention attribute on %s",
                          type(target).__name__)
            return None

    return new_block


__all__ = [
    "Jamba2Block",
    "SVDMambaSub",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_jamba",
]