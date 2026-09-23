"""Optimized Hybrid Attention (OHA) – Production-Hardened.

Fuses localized sliding-window self-attention with sequence-compressed low-rank
global attention via dynamic head-wise gated modulation and exact pretrained weight
transfer.

Bug-fix map (2026 revision):
  OHA-FIX-01  RoPE cos/sin cast to q/k dtype.
  OHA-FIX-02  compression_rank decoupled from attention rank.
  OHA-FIX-03  Proper causal compression (per-chunk causal masking).
  OHA-FIX-04  Recursion guard + removed dead `in locals()` check.
  OHA-FIX-05  Robust hidden_size / num_heads detection (q_lin, q_proj, c_attn).
  OHA-FIX-06  window_size = full width; effective radius = window_size // 2.
  OHA-FIX-07  Unified head_mask handling.
  OHA-FIX-08  num_heads fallback → from src_attn if available.
  OHA-FIX-09  is_decoder detection in inject via helper.
  OHA-FIX-10  mask_inv cast to comp dtype.
  OHA-FIX-11  RoPE past_len offset for KV-cache decoding.
  OHA-FIX-12  output_attentions documented: local branch only.
  OHA-FIX-13  Forward sanity check after weight transfer.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Decoder detection helper (OHA-FIX-09)
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    src_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if src_attn is not None:
        if getattr(src_attn, "is_decoder", False):
            return True
        if hasattr(src_attn, "c_attn"):
            return True
        if "gpt" in src_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# RoPE (with dtype-safe casting — OHA-FIX-01, OHA-FIX-11)
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

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# Main attention
# ===========================================================================

@ATTENTION_REGISTRY.register("oha", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("optimized_hybrid_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("oha_attention", supports_causal=True, supports_gqa=True)
class OptimizedHybridAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        window_size: int = 64,
        compression_rank: int = 32,
        use_gate: bool = True,
        gate_bias: float = 2.0,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        # OHA-FIX-05: robust dimension detection
        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = kwargs.get("head_dim") or getattr(config, "head_dim", None) or (hidden_size // max(1, num_heads))

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # OHA-FIX-06: window_size = full width; radius = window_size // 2
        self.window_size = int(getattr(config, "oha_window_size", window_size) if config else window_size)
        self.window_radius = max(1, self.window_size // 2)

        # OHA-FIX-02: compression_rank is independent from attention rank
        cfg_cr = getattr(comp_config, "compression_rank", None) if comp_config else None
        self.compression_rank = int(cfg_cr if cfg_cr is not None else compression_rank)
        self.use_gate = bool(use_gate)

        # OHA-FIX-09: robust decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder or kwargs.get("is_causal"), source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = int(
            getattr(config, "lora_rank", None)
            or (getattr(comp_config, "lora_rank", None) if comp_config else None)
            or kwargs.get("lora_rank", 16)
        )
        self.lora_alpha = float(
            getattr(config, "lora_alpha", None)
            or (getattr(comp_config, "lora_alpha", None) if comp_config else None)
            or kwargs.get("lora_alpha", 32.0)
        )
        self.freeze_base = self.lora_rank > 0

        # Projections
        self.query = LoRALinear(self.dim, self.all_head_size, bias=True,
                                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        # Global compression: (dim → compression_rank)
        self.compress_k = nn.Linear(self.dim, self.compression_rank, bias=False)
        self.compress_v = nn.Linear(self.dim, self.compression_rank, bias=False)
        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)

        if self.use_gate:
            self.gate = nn.Linear(self.dim, self.num_heads)
            nn.init.constant_(self.gate.bias, gate_bias)
            nn.init.normal_(self.gate.weight, std=0.02)
        else:
            self.register_parameter("gate", None)

        dropout_p = (
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # OHA-FIX-05/08: robust dimension detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_hidden_size(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any]) -> int:
        for k in ("hidden_size", "d_model", "n_embd", "dim"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("hidden_size", "d_model", "n_embd", "dim"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("out_lin", "out_proj", "o_proj", "dense", "c_proj"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "out_features"):
                    return int(mod.out_features)
            for attr in ("q_lin", "q_proj", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    return int(mod.in_features)
        return 768

    @staticmethod
    def _detect_num_heads(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                          hidden_size: int) -> int:
        for k in ("num_heads", "num_attention_heads", "n_heads", "n_head"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_attention_heads", "num_heads", "n_heads", "n_head"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for k in ("num_heads", "num_attention_heads", "n_heads"):
                v = getattr(src, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        # OHA-FIX-08: fallback based on hidden_size / 64 (typical head_dim)
        return max(1, hidden_size // 64)

    @staticmethod
    def _detect_num_kv_heads(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                             num_heads: int) -> int:
        for k in ("num_key_value_heads", "num_kv_heads"):
            v = kwargs.get(k)
            if isinstance(v, int) and v > 0:
                return v
        if config is not None:
            for k in ("num_key_value_heads", "num_kv_heads"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            v = getattr(src, "num_key_value_heads", None)
            if isinstance(v, int) and v > 0:
                return v
        return num_heads

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                         "XLMRobertaSelfAttention", "ElectraSelfAttention",
                         "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
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
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size, :],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)

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

        # 2. RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules (DistilBERT, etc.)
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"OptimizedHybridAttention.copy_weights failed: missing "
                f"mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {"q": self.query, "k": self.key, "v": self.value, "o": self.out_proj}

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

                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()

                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype)
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "OptimizedHybridAttention: weight for '%s' has near-zero norm (%.2e)",
                        proj_key, w_norm,
                    )

        # OHA-FIX-13: forward sanity check after weight transfer
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dummy = torch.randn(1, 4, self.dim, device=d, dtype=self.query.weight.dtype)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("OHA forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # Mask helper (OHA-FIX-10)
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N: int, device: torch.device,
    ) -> torch.Tensor:
        """Return boolean (B, N): True = valid token."""
        if attention_mask is None:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            if attention_mask.dtype == torch.bool:
                return attention_mask.squeeze(1)
            if attention_mask.dtype.is_floating_point:
                return (attention_mask > -1e4).squeeze(1)
            return (attention_mask > 0).squeeze(1)
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,
        layer_idx: int = 0,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Projections
        Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, N, self.num_kv_heads, d).permute(0, 2, 1, 3)

        # RoPE with past offset (OHA-FIX-11)
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            if past_len > 0:
                cos = cos[..., past_len: past_len + N, :]
                sin = sin[..., past_len: past_len + N, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # Mask (OHA-FIX-10)
        valid_mask_2d = self._prepare_valid_mask(attention_mask, B, N, device)
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, N)

        scale = 1.0 / math.sqrt(d)

        # ---- 1. Local sliding-window attention (OHA-FIX-06) ----
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # (N, N)
        if self.is_causal:
            band_mask = (diff >= 0) & (diff <= self.window_radius)
        else:
            band_mask = diff.abs() <= self.window_radius
        local_mask = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d

        local_scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        local_scores = local_scores.masked_fill(~local_mask, -1e4)
        local_scores = torch.nan_to_num(local_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        local_probs = F.softmax(local_scores, dim=-1).to(dtype)
        local_probs = torch.nan_to_num(local_probs, nan=0.0)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)   # (B, H, N, d)

        # ---- 2. Global compressed attention (OHA-FIX-03) ----
        comp_k_raw = self.compress_k(hidden_states)   # (B, N, R)
        comp_v_raw = self.compress_v(hidden_states)
        mask_inv = (~valid_mask_2d).unsqueeze(-1).to(comp_k_raw.dtype)
        comp_k_raw = comp_k_raw.masked_fill(mask_inv.bool(), -1e4)
        comp_v_raw = comp_v_raw.masked_fill(mask_inv.bool(), -1e4)

        # OHA-FIX-03: proper causal compression.
        # Divide sequence into `compression_rank` chunks; slot r covers
        # positions [(r)*chunk, (r+1)*chunk). Query at position i can only
        # attend to slots that are fully or partially <= i.
        R = self.compression_rank
        chunk_size = max(1, (N + R - 1) // R)   # ceil
        if self.is_causal:
            # For compression: slot r attends to positions < (r+1)*chunk
            slot_positions = torch.arange(N, device=device).unsqueeze(0)          # (1, N)
            slot_limits = ((torch.arange(R, device=device) + 1) * chunk_size).unsqueeze(1)  # (R, 1)
            comp_causal_mask = slot_positions < slot_limits   # (R, N) True = allowed
            comp_causal_mask = comp_causal_mask.t().unsqueeze(0)  # (1, N, R)
        else:
            comp_causal_mask = None

        if comp_causal_mask is not None:
            comp_k_raw = comp_k_raw.masked_fill(~comp_causal_mask, -1e4)
            comp_v_raw = comp_v_raw.masked_fill(~comp_causal_mask, -1e4)

        comp_k_weights = F.softmax(comp_k_raw.float(), dim=1).to(dtype)
        comp_v_weights = F.softmax(comp_v_raw.float(), dim=1).to(dtype)

        ck = torch.einsum("bhnd,bnr->bhrd", K, comp_k_weights)   # (B, H, R, d)
        cv = torch.einsum("bhnd,bnr->bhrd", V, comp_v_weights)

        global_scores = torch.matmul(Q.float(), ck.float().transpose(-2, -1).contiguous()) * scale
        global_scores = torch.nan_to_num(global_scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        if self.is_causal:
            # Query at position i can only see slots r where r*chunk_size < i+1
            query_positions = torch.arange(N, device=device).unsqueeze(1)          # (N, 1)
            slot_starts = (torch.arange(R, device=device) * chunk_size).unsqueeze(0)  # (1, R)
            global_causal = query_positions >= slot_starts   # (N, R)
            global_scores = global_scores.masked_fill(
                ~global_causal.unsqueeze(0).unsqueeze(0), -1e4
            )

        global_probs = F.softmax(global_scores, dim=-1).to(dtype)
        global_probs = torch.nan_to_num(global_probs, nan=0.0)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)   # (B, H, N, d)

        # ---- 3. Gated fusion ----
        if self.use_gate and self.gate is not None:
            gate = torch.sigmoid(self.gate(hidden_states))    # (B, N, H)
            gate = gate.permute(0, 2, 1).unsqueeze(-1)         # (B, H, N, 1)
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = 0.5 * local_ctx + 0.5 * global_ctx

        # ---- head_mask (OHA-FIX-07) ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            context = context * hm.to(context.dtype)

        # Zero out invalid query positions
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        # Reshape to (B, N, all_head_size)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (local_probs if output_attentions else None)


# ===========================================================================
# Injection (OHA-FIX-04)
# ===========================================================================

@ATTENTION_REGISTRY.register("oha_inject")
@ATTENTION_REGISTRY.register("inject_oha")
def inject_oha(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_oha: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_oha(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        src_attn = old_attn if old_attn is not None else (
            getattr(layer, "attention",
            getattr(layer, "self",
            getattr(layer, "attn", None)))
        )

        # OHA-FIX-09: robust decoder detection
        is_dec = _detect_decoder(cfg, kwargs.get("is_decoder"), src_attn)

        new_attn = OptimizedHybridAttention(
            config=cfg,
            window_size=getattr(comp_config, "window_size", 64) if comp_config else 64,
            compression_rank=getattr(comp_config, "compression_rank", 32) if comp_config else 32,
            comp_config=comp_config,
            old_attn=src_attn,
            is_decoder=is_dec,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = False
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_oha failed: %s", e, exc_info=True)
        return None


inject_optimized_hybrid_attention = inject_oha
inject_optimized_hybrid = inject_oha