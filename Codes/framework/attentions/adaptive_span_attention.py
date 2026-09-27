"""
Adaptive Span Attention — Final Corrected Implementation.
Based on: Sukhbaatar et al., 2019 (arXiv:1905.07799)

Math:
    z_h = sigmoid(logit_h) * S_max
    g_{z_h}(d) = clamp((R + z_h - d) / R, 0, 1)
    A = Softmax(S + M) ⊙ g_z   (then renormalize)
    O = (A V) W_O

Notes on the ramp application order:
    Multiplying after softmax and renormalizing is *mathematically equivalent*
    to the paper's "mask inside softmax" formulation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter


# =============================================================================
# Loss helpers
# =============================================================================

def get_total_span_loss(model: nn.Module, lambda_span: float = 0.01) -> torch.Tensor:
    r"""Total L1 span regularization loss across all AdaptiveSpanAttention modules.

    L_span = lambda * (1/|M|) * sum_m mean_h(sigmoid(logit_mh))
    """
    span_modules = getattr(model, "_adaptive_span_modules", None)
    if span_modules is None:
        span_modules = [m for m in model.modules() if isinstance(m, AdaptiveSpanAttention)]

    losses: List[torch.Tensor] = []
    for m in span_modules:
        if hasattr(m, "get_span_regularization_loss"):
            losses.append(m.get_span_regularization_loss(lambda_span=lambda_span))

    if not losses:
        # Attach to the model's graph so backward does not fail on a leaf tensor.
        try:
            any_param = next(model.parameters())
            return any_param.new_zeros(())
        except StopIteration:
            return torch.zeros((), dtype=torch.float32)

    return torch.stack(losses).mean()


# =============================================================================
# RoPE helpers
# =============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _gather_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather cos/sin rows using absolute position_ids (supports cache offsets)."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    Bc = cos.shape[0]
    S = cos.shape[-2]
    d = cos.shape[-1]

    ids = position_ids.clamp(min=0, max=S - 1)  # [B, N]

    if Bc == 1:
        cos_g = cos[0, 0][ids]  # [B, N, d]
        sin_g = sin[0, 0][ids]
        cos_g = cos_g.unsqueeze(1)  # [B, 1, N, d]
        sin_g = sin_g.unsqueeze(1)
    else:
        cos_g = torch.gather(
            cos.expand(-1, -1, -1, -1).squeeze(1),
            1,
            ids.unsqueeze(-1).expand(-1, -1, d),
        ).unsqueeze(1)
        sin_g = torch.gather(
            sin.expand(-1, -1, -1, -1).squeeze(1),
            1,
            ids.unsqueeze(-1).expand(-1, -1, d),
        ).unsqueeze(1)
    return cos_g, sin_g


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using absolute position_ids."""
    if cos is None or sin is None:
        return q, k

    N = q.shape[-2]

    if position_ids is not None:
        cos, sin = _gather_cos_sin(cos, sin, position_ids)
    else:
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        if cos.shape[-2] != N:
            raise ValueError(
                f"RoPE cos length ({cos.shape[-2]}) does not match sequence ({N}) "
                f"and position_ids was not provided."
            )

    head_dim = q.shape[-1]
    rotary_dim = cos.shape[-1]

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


# =============================================================================
# Mask utilities
# =============================================================================

def _safe_mask_value(dtype: torch.dtype) -> float:
    """Return a numerically safe mask value for the given dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


def _build_keep_mask(
    B: int,
    N_q: int,
    N_k: int,
    is_causal: bool,
    attention_mask: Optional[torch.Tensor],
    encoder_attention_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a boolean mask [B, 1, N_q, N_k] where True = keep."""
    mask = None

    if is_causal:
        if N_q == N_k:
            causal = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool))
        else:
            offset = N_k - N_q
            q_idx = torch.arange(N_q, device=device).unsqueeze(1) + offset
            k_idx = torch.arange(N_k, device=device).unsqueeze(0)
            causal = k_idx <= q_idx
        mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_k]

    def _combine(am: torch.Tensor) -> torch.Tensor:
        nonlocal mask
        if am.dim() == 2:  # [B, N_k]
            am_b = am[:, None, None, :]
        elif am.dim() == 3:  # [B, 1, N_k] or [B, N_q, N_k]
            am_b = am.unsqueeze(1) if am.shape[1] != N_q else am[:, None, :, :]
        elif am.dim() == 4:
            am_b = am
        else:
            return mask if mask is not None else torch.ones((B, 1, N_q, N_k), device=device, dtype=torch.bool)

        if am_b.dtype != torch.bool:
            # Distinguish 0/1 mask from additive bias
            if am_b.max() <= 1.0 and am_b.min() >= 0.0:
                am_b = am_b.bool()
            else:
                am_b = am_b > -1e3  # heuristic for additive masks
        return am_b

    if attention_mask is not None:
        am = _combine(attention_mask)
        mask = am if mask is None else (mask & am)

    if encoder_attention_mask is not None:
        em = _combine(encoder_attention_mask)
        mask = em if mask is None else (mask & em)

    return mask


# =============================================================================
# LoRA Linear
# =============================================================================

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
            fan_in = self.weight.shape[1]
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
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


class IdentityOutProj(nn.Identity):
    """Identity projection with LoRA-compatible interface (instance attributes)."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_A = None
        self.lora_B = None

    def freeze_base_weights(self) -> None:
        pass


# =============================================================================
# Adaptive Span Attention
# =============================================================================

@ATTENTION_REGISTRY.register("adaptive_span")
@ATTENTION_REGISTRY.register("adaptive_span_attention")
class AdaptiveSpanAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = False
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        max_span: int = 128,
        ramp_size: float = 32.0,
        lora_rank: int = 16,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        layer_idx: Optional[int] = None,
        return_style: Optional[str] = None,
        rolling_cache_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # ---------------- Dimensions ----------------
        hidden_size = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
        num_heads = kwargs.get("num_heads") or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")
        head_dim = 0

        if config is not None:
            adapter = ModelAdapter(config)
            hidden_size = hidden_size or adapter.hidden_size
            num_heads = num_heads or adapter.num_heads
            num_kv_heads = num_kv_heads or adapter.num_key_value_heads
            head_dim = getattr(adapter, "head_dim", 0)

        if not hidden_size and source_attn is not None:
            cand = getattr(source_attn, "hidden_size",
                    getattr(source_attn, "embed_dim",
                    getattr(source_attn, "all_head_size", None)))
            if cand is None and hasattr(source_attn, "q_proj"):
                cand = getattr(source_attn.q_proj, "in_features", None)
            hidden_size = cand

        hidden_size = int(hidden_size or 768)
        num_heads = int(num_heads or 12)
        num_kv_heads = int(num_kv_heads or num_heads)
        head_dim = head_dim if head_dim > 0 else (hidden_size // num_heads if num_heads > 0 else 64)

        self.hidden_size: int = hidden_size
        self.d_model: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.layer_idx: int = int(layer_idx) if layer_idx is not None else 0
        self.rolling_cache_size: Optional[int] = rolling_cache_size

        # ---------------- Span config ----------------
        if config is not None:
            self.max_span = int(getattr(config, "max_span", max_span))
            self.ramp_size = float(getattr(config, "ramp_size", ramp_size))
        elif comp_config is not None:
            self.max_span = int(getattr(comp_config, "max_span", max_span))
            self.ramp_size = float(getattr(comp_config, "ramp_size", ramp_size))
        else:
            self.max_span = int(max_span)
            self.ramp_size = float(ramp_size)

        # ---------------- Decoder / nested encoder detection ----------------
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = getattr(config, "is_decoder",
                         getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        _dec_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
                         "bloom", "falcon", "olmo", "mpt", "stablelm")
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
            source_cls in ("BertSelfAttention", "RobertaSelfAttention")
            or cfg_model_type in ("roberta", "bert")
            or (source_cls.endswith(("SelfAttention", "SelfAttn"))
                and "distil" not in source_cls.lower()
                and "gpt" not in source_cls.lower()
                and "qwen" not in source_cls.lower())
        )
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", is_nested))

        # ---------------- Return-style auto-detection ----------------
        if return_style is None:
            if self.is_nested_encoder:
                return_style = "bert"
            elif "GPT2" in source_cls or "gpt2" in cfg_model_type:
                return_style = "gpt2"
            elif "T5" in source_cls or "t5" in cfg_model_type:
                return_style = "t5"
            elif self.is_decoder:
                return_style = "llama"  # modern decoder default
            else:
                return_style = "bert"
        self.return_style: str = str(return_style)

        # ---------------- T5 relative position bias ----------------
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32) if config else 32
        self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128) if config else 128

        # ---------------- LoRA config ----------------
        if config is not None:
            resolved_lora_rank = getattr(config, "lora_rank",
                                  getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)))
            resolved_lora_alpha = getattr(config, "lora_alpha",
                                   getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            resolved_lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank))
            resolved_lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            resolved_lora_rank = kwargs.get("lora_rank", lora_rank)
            resolved_lora_alpha = kwargs.get("lora_alpha", 32.0)

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            for p in source_attn.parameters():
                target_dtype = p.dtype
                break

        self.lora_rank = int(resolved_lora_rank)
        self.lora_alpha = float(resolved_lora_alpha)
        self.freeze_base: bool = (self.lora_rank > 0)

        # ---------------- Projections ----------------
        self.q_proj = LoRALinear(self.hidden_size, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base, dtype=target_dtype)
        self.k_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base, dtype=target_dtype)
        self.v_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base, dtype=target_dtype)
        if self.is_nested_encoder:
            self.out_proj: nn.Module = IdentityOutProj()
        else:
            self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True,
                                       lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                       freeze_base=self.freeze_base, dtype=target_dtype)

        # =========================================================================
        # FIX: Zero-Shot Readiness
        # Changed initialization from -2.0 to 5.0 so that sigmoid(5.0) ~ 0.99
        # This gives the model a near-full context window for zero-shot eval
        # preventing the "15-token amnesia" bug.
        # =========================================================================
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), 5.0, dtype=torch.float32))

        dropout_p = getattr(config, "dropout",
                    getattr(config, "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", dropout))) if config else dropout
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.q_proj.to(target_dtype)
            self.k_proj.to(target_dtype)
            self.v_proj.to(target_dtype)
            self.out_proj.to(target_dtype)

    # -------------------------------------------------------------------------
    # Span losses
    # -------------------------------------------------------------------------

    @property
    def current_span(self) -> torch.Tensor:
        """Per-head span in tokens. Shape: [H]."""
        return torch.sigmoid(self.span_logits) * self.max_span

    def get_span_regularization_loss(self, lambda_span: float = 0.01) -> torch.Tensor:
        """lambda * mean_h(sigmoid(logit_h)) — normalized to [0,1]."""
        return lambda_span * torch.sigmoid(self.span_logits).mean()

    # -------------------------------------------------------------------------
    # T5 relative position bias
    # -------------------------------------------------------------------------

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
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
            rel, bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.permute([2, 0, 1]).unsqueeze(0)

    # -------------------------------------------------------------------------
    # Weight copy
    # -------------------------------------------------------------------------

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback for %s: %s", proj, e)

        # GPT-2 fused Conv1D
        if hasattr(src, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c_attn_layer = src.c_attn
            base_c = getattr(c_attn_layer, "base_layer", c_attn_layer)
            weight = getattr(base_c, "weight", getattr(c_attn_layer, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_layer, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    hasattr(c_attn_layer, "nf") or hasattr(base_c, "nf")
                    or c_attn_layer.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                )
                if is_conv1d and weight.dim() == 2:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                extracted["q"] = (weight[:split].contiguous(),
                                  bias_data[:split].contiguous() if bias_data is not None else None)
                extracted["k"] = (weight[split:2 * split].contiguous(),
                                  bias_data[split:2 * split].contiguous() if bias_data is not None else None)
                extracted["v"] = (weight[2 * split:].contiguous(),
                                  bias_data[2 * split:].contiguous() if bias_data is not None else None)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_layer = src.c_proj
            base_o = getattr(c_proj_layer, "base_layer", c_proj_layer)
            w_o = getattr(base_o, "weight", getattr(c_proj_layer, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    hasattr(c_proj_layer, "nf") or hasattr(base_o, "nf")
                    or c_proj_layer.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                )
                if is_conv1d and w_o.dim() == 2:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_layer, "bias", None))
                extracted["o"] = (w_o.contiguous(),
                                  b_o.data.clone().contiguous() if b_o is not None else None)

        # BERT/RoBERTa nested
        if hasattr(src, "self") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and getattr(src, "output", None) is not None \
                and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # Named projections
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

        # Confirm nested-encoder
        cls_name = src.__class__.__name__
        cfg_type = getattr(self.config, "model_type", "").lower() if self.config else ""
        if not self.is_nested_encoder:
            self.is_nested_encoder = bool(
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or cfg_type in ("roberta", "bert")
                or (cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                    and "gpt" not in cls_name.lower()
                    and "qwen" not in cls_name.lower())
            )
        if self.is_nested_encoder and not isinstance(self.out_proj, IdentityOutProj):
            self.out_proj = IdentityOutProj()

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("AdaptiveSpanAttention missing weights for projections: %s", missing)

        with torch.no_grad():
            for proj_key, target in (("q", self.q_proj), ("k", self.k_proj),
                                     ("v", self.v_proj), ("o", self.out_proj)):
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
                target.freeze_base_weights()

        if getattr(src, "rotary_emb", None) is not None:
            self.rotary_emb = src.rotary_emb
        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if getattr(src, "relative_attention_bias", None) is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

        # ---------------- QKV ----------------
        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, -1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, -1, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # ---------------- Position ids (with cache offset) ----------------
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None and not is_cross_attention:
            past_len = 0
            if past_key_value is not None and past_key_value is not False:
                if hasattr(past_key_value, "get_seq_length"):
                    try:
                        past_len = past_key_value.get_seq_length(self.layer_idx)
                    except (TypeError, AttributeError):
                        try:
                            past_len = past_key_value.get_seq_length()
                        except Exception:
                            past_len = 0
                elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2 \
                        and past_key_value[0] is not None:
                    past_len = past_key_value[0].shape[-2]
            position_ids = torch.arange(
                past_len, past_len + N_q, device=device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)

        # ---------------- RoPE ----------------
        pos_emb = position_embeddings if position_embeddings is not None \
                  else kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin, position_ids=position_ids)
        elif self.rotary_emb is not None:
            if position_ids is None:
                # Fallback for cross-attention: keys have their own positions
                position_ids = torch.arange(N_q, device=device, dtype=torch.long)\
                                    .unsqueeze(0).expand(B, -1)
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

        # ---------------- KV cache ----------------
        new_past_key_value: Optional[Any] = None
        use_cache = kwargs.get("use_cache", False) or past_key_value is not None

        if is_cross_attention:
            # Cross-attention: no cache update
            K, V = K_new, V_new
            new_past_key_value = past_key_value  # pass through
        else:
            if past_key_value is not None and past_key_value is not False:
                if hasattr(past_key_value, "update"):  # HF DynamicCache
                    try:
                        K, V, new_past_key_value = past_key_value.update(K_new, V_new, self.layer_idx)
                    except Exception as e:
                        logger.debug("DynamicCache.update failed: %s. Falling back to tuple.", e)
                        K = torch.cat([past_key_value[0], K_new], dim=-2)
                        V = torch.cat([past_key_value[1], V_new], dim=-2)
                        new_past_key_value = (K, V)
                elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                    past_k, past_v = past_key_value[0], past_key_value[1]
                    if past_k is not None:
                        K = torch.cat([past_k, K_new], dim=-2)
                        V = torch.cat([past_v, V_new], dim=-2)
                    else:
                        K, V = K_new, V_new
                    new_past_key_value = (K, V)
                else:
                    K, V = K_new, V_new
                    new_past_key_value = (K, V)
            else:
                K, V = K_new, V_new
                if use_cache:
                    new_past_key_value = (K, V)

            # Rolling buffer (optional): keep only the last `rolling_cache_size` tokens.
            if self.rolling_cache_size is not None and K.shape[-2] > self.rolling_cache_size:
                K = K[..., -self.rolling_cache_size:, :].contiguous()
                V = V[..., -self.rolling_cache_size:, :].contiguous()
                if isinstance(new_past_key_value, tuple):
                    new_past_key_value = (K, V)

        N_k = K.shape[-2]

        # ---------------- GQA repeat (with double-repeat guard) ----------------
        if self.kv_repeat > 1:
            if K.shape[1] == self.num_kv_heads:
                K_full = repeat_kv(K, self.kv_repeat)
                V_full = repeat_kv(V, self.kv_repeat)
            elif K.shape[1] == self.num_heads:
                # Already repeated (e.g. user-provided cache)
                K_full, V_full = K, V
            else:
                raise ValueError(
                    f"Unexpected K head dim {K.shape[1]} (expected {self.num_kv_heads} or {self.num_heads})."
                )
        else:
            K_full, V_full = K, V

        # ---------------- Scores ----------------
        scale = d ** -0.5
        scores = torch.matmul(Q, K_full.transpose(-2, -1).contiguous()) * scale

        # ---------------- T5 position bias ----------------
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=device)
        if position_bias is not None:
            scores = scores + position_bias.to(scores.dtype)

        # ---------------- Combined boolean mask ----------------
        is_causal = self.is_decoder and not is_cross_attention
        keep_mask = _build_keep_mask(
            B=B, N_q=N_q, N_k=N_k,
            is_causal=is_causal,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask if is_cross_attention else None,
            device=device,
        )

        # Track fully-masked rows (for safe softmax)
        fully_masked: Optional[torch.Tensor] = None

        if keep_mask is not None:
            fully_masked = ~keep_mask.any(dim=-1, keepdim=True)  # [B, 1, N_q, 1]
            mask_val = _safe_mask_value(scores.dtype)
            scores = scores.masked_fill(~keep_mask, mask_val)

        # ---------------- Softmax (safe) ----------------
        if fully_masked is not None:
            scores = torch.where(fully_masked, torch.zeros_like(scores), scores)

        if dtype in (torch.float16, torch.bfloat16) and scores.numel() > 2 ** 30:
            attn = torch.empty_like(scores)
            chunk = max(64, min(512, N_q))
            for s in range(0, N_q, chunk):
                e = min(s + chunk, N_q)
                attn[:, :, s:e, :] = F.softmax(scores[:, :, s:e, :].float(), dim=-1).to(dtype)
        else:
            attn = F.softmax(scores.float(), dim=-1).to(dtype)

        if fully_masked is not None:
            attn = attn.masked_fill(fully_masked, 0.0)

        # ---------------- Adaptive span ramp ----------------
        # Only for self-attention. Cross-attention has no notion of "distance".
        if not is_cross_attention:
            span = torch.sigmoid(self.span_logits) * self.max_span  # [H]

            # Position indices (absolute for RoPE-aligned distance)
            if position_ids is not None:
                # position_ids: [B, N_q]; pad with past offset if using cache with rolling
                q_pos = position_ids[0].to(torch.float32)  # assume shared across batch
                if N_k == N_q:
                    k_pos = q_pos
                else:
                    last_q = q_pos[-1].item()
                    k_pos = torch.arange(last_q - N_k + 1, last_q + 1,
                                         device=device, dtype=torch.float32)
            else:
                q_pos = torch.arange(N_q, device=device, dtype=torch.float32)
                k_pos = torch.arange(N_k, device=device, dtype=torch.float32)

            if is_causal:
                # For generation offset: q_pos[i] - k_pos[j] with clamp>=0
                dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).clamp(min=0.0)
            else:
                dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()

            # ramp: [H, N_q, N_k] (no batch dim -> memory O(H*N^2), not O(B*H*N^2))
            span_h = span.view(H, 1, 1).to(torch.float32)
            dist_mat = dist.unsqueeze(0)  # [1, N_q, N_k]
            ramp = torch.clamp(
                (self.ramp_size + span_h - dist_mat) / self.ramp_size,
                0.0, 1.0,
            ).to(attn.dtype)  # [H, N_q, N_k]

            # Broadcast against attn [B, H, N_q, N_k]
            attn = attn * ramp.unsqueeze(0)
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn = torch.where(attn_sum > 1e-6,
                               attn / attn_sum.clamp(min=1e-6),
                               torch.zeros_like(attn))
            if fully_masked is not None:
                attn = attn.masked_fill(fully_masked, 0.0)

        # ---------------- Dropout + head mask ----------------
        attn = self.dropout(attn)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        # ---------------- Context ----------------
        context = torch.matmul(attn, V_full)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        # ---------------- Return (style-driven) ----------------
        return self._format_output(output, attn if output_attentions else None,
                                   new_past_key_value, position_bias, output_attentions)

    # -------------------------------------------------------------------------
    # Output formatting
    # -------------------------------------------------------------------------

    def _format_output(
        self,
        output: torch.Tensor,
        attn: Optional[torch.Tensor],
        past_kv: Optional[Any],
        position_bias: Optional[torch.Tensor],
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

        return output, attn, past_kv


# =============================================================================
# Injection factory
# =============================================================================

@ATTENTION_REGISTRY.register("adaptive_span_inject")
def inject_adaptive_span(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[AdaptiveSpanAttention, nn.Module]]:
    """Inject AdaptiveSpanAttention into a pretrained transformer (or a single layer)."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1) Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        injected: List[nn.Module] = []
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_span(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, **kwargs,
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

    # 2) Single layer
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

    def _already_wrapped(obj: Any) -> bool:
        if obj is None:
            return False
        if isinstance(obj, AdaptiveSpanAttention):
            return True
        if "AdaptiveSpan" in obj.__class__.__name__:
            return True
        inner = getattr(obj, "inner", None)
        wrapped = getattr(obj, "_wrapped_attention", None)
        return isinstance(inner, AdaptiveSpanAttention) or isinstance(wrapped, AdaptiveSpanAttention)

    if _already_wrapped(src_attn):
        return None
    for attr in ("attention", "self_attn", "attn", "self"):
        if _already_wrapped(getattr(target, attr, None)):
            return None

    if src_attn is not None:
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (src_attn.q_proj.in_features if hasattr(src_attn, "q_proj")
            and hasattr(src_attn.q_proj, "in_features") else None)
        or (src_attn.q_lin.in_features if hasattr(src_attn, "q_lin")
            and hasattr(src_attn.q_lin, "in_features") else None)
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
        (hasattr(target, "attention") and hasattr(target.attention, "output")
         and hasattr(target.attention.output, "dense"))
        or (hasattr(target, "output") and hasattr(target.output, "dense"))
    )

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
            is_nested_encoder=(True if is_nested_layer else kwargs.get("is_nested_encoder", None)),
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception:
        logger.exception("AdaptiveSpanAttention construction failed.")
        return None

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