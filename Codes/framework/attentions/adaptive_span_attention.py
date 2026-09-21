r"""
Adaptive Span Attention Module (Sukhbaatar et al., 2019).

Mathematical Formulation:
    1. Query-Key Attention Scores:
       $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$

    2. Dynamic Per-Head Span Masking (Continuous Soft Ramp):
       Let $z_h = \sigma(\text{logit}_h) \cdot S_{\max}$, and $R$ be ramp size.
       $$g_{z_h}(d) = \operatorname{clamp}\left(\frac{R + z_h - d}{R}, 0, 1\right)$$

    3. Context Aggregation:
       $$A = \operatorname{Softmax}(S + M_{\text{causal}} + M_{\text{pad}}) \odot g_z$$
       $$O = (A V) W_O$$
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter


def get_total_span_loss(model: nn.Module, lambda_span: float = 0.01) -> torch.Tensor:
    r"""Compute total L1 span regularization loss across all AdaptiveSpanAttention modules in model.

    L_span = \lambda \cdot \frac{1}{|M|} \sum_{m \in M} \frac{1}{H} \sum_{h=1}^H \frac{S_h}{S_{\max}}
    """
    span_losses = []
    # Check cached list or search model modules
    span_modules = getattr(model, "_adaptive_span_modules", None)
    if span_modules is None:
        span_modules = [m for m in model.modules() if isinstance(m, AdaptiveSpanAttention)]

    for m in span_modules:
        if hasattr(m, "get_span_regularization_loss"):
            span_losses.append(m.get_span_regularization_loss(lambda_span=lambda_span))
        elif hasattr(m, "current_span") and hasattr(m, "max_span"):
            # Fallback normalized span loss
            norm_span = m.current_span / max(1, m.max_span)
            span_losses.append(lambda_span * norm_span.mean())

    if not span_losses:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(span_losses).mean()


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

    seq_len = q.shape[-2]
    if cos.shape[-2] < seq_len:
        # Extend cos and sin along sequence dim by repeating the last slice
        diff = seq_len - cos.shape[-2]
        last_cos = cos[..., -1:, :]
        last_sin = sin[..., -1:, :]
        cos = torch.cat([cos, last_cos.expand(*cos.shape[:-2], diff, cos.shape[-1])], dim=-2)
        sin = torch.cat([sin, last_sin.expand(*sin.shape[:-2], diff, sin.shape[-1])], dim=-2)
    elif cos.shape[-2] > seq_len:
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]

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
    lora_A = None
    lora_B = None

    def freeze_base_weights(self) -> None:
        pass


@ATTENTION_REGISTRY.register("adaptive_span", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_span_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class AdaptiveSpanAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        max_span: int = 128,
        lora_rank: int = 16,
        comp_config: Optional[Any] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # Resolve hidden_size and num_heads safely even if config is None
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
            cand = getattr(source_attn, "hidden_size", getattr(source_attn, "embed_dim", getattr(source_attn, "all_head_size", None)))
            if cand is None and hasattr(source_attn, "q_proj") and hasattr(source_attn.q_proj, "in_features"):
                cand = source_attn.q_proj.in_features
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

        if config is not None:
            self.max_span = int(getattr(config, "max_span", max_span))
            self.ramp_size = float(getattr(config, "ramp_size", kwargs.get("ramp_size", 32.0)))
        elif comp_config is not None:
            self.max_span = int(getattr(comp_config, "max_span", max_span))
            self.ramp_size = float(getattr(comp_config, "ramp_size", kwargs.get("ramp_size", 32.0)))
        else:
            self.max_span = int(max_span)
            self.ramp_size = float(kwargs.get("ramp_size", 32.0))

        model_type = getattr(config, "model_type", "").lower() if config else ""

        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config else False
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)

        # Detect nested encoder architecture early
        source_cls = source_attn.__class__.__name__ if source_attn is not None else ""
        cfg_model_type = getattr(config, "model_type", "").lower() if config else ""
        is_nested = (
            source_cls in ("BertSelfAttention", "RobertaSelfAttention")
            or cfg_model_type in ("roberta", "bert")
            or (source_cls.endswith(("SelfAttention", "SelfAttn")) and "distil" not in source_cls.lower() and "gpt" not in source_cls.lower() and "qwen" not in source_cls.lower())
        )
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", is_nested))
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None

        if config is not None:
            self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32)
            self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128)
            resolved_lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)))
            resolved_lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            resolved_lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank))
            resolved_lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            resolved_lora_rank = kwargs.get("lora_rank", lora_rank)
            resolved_lora_alpha = kwargs.get("lora_alpha", 32.0)

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            target_dtype = next(source_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        self.lora_rank = int(resolved_lora_rank)
        self.lora_alpha = float(resolved_lora_alpha)
        self.freeze_base: bool = (self.lora_rank > 0)

        self.q_proj = LoRALinear(self.hidden_size, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.k_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.v_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        out_lora_rank = 0 if self.is_nested_encoder else self.lora_rank
        if self.is_nested_encoder:
            self.out_proj = IdentityOutProj()
        else:
            self.out_proj = LoRALinear(self.all_head_size, self.hidden_size, bias=True, lora_rank=out_lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)

        # Initialize to 4.0 so span starts almost fully open (sigmoid(4.0) ~ 0.982)
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), 4.0, dtype=torch.float32))

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", dropout),
            ),
        ) if config else dropout
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn)
            
        if target_dtype is not None:
            self.q_proj.to(target_dtype)
            self.k_proj.to(target_dtype)
            self.v_proj.to(target_dtype)
            self.out_proj.to(target_dtype)

        if hasattr(self, "source_attn"):
            del self.source_attn

    @property
    def current_span(self) -> torch.Tensor:
        """Returns the current learned span (in number of tokens) for each head."""
        return torch.sigmoid(self.span_logits) * self.max_span

    def get_span_loss(self) -> torch.Tensor:
        """L1 regularization penalty on the span to encourage context pruning as in Sukhbaatar et al. (2019)."""
        return self.current_span.sum()

    def get_span_regularization_loss(self, lambda_span: float = 0.01) -> torch.Tensor:
        r"""Compute normalized per-head L1 span penalty:
        \lambda \cdot \frac{1}{H} \sum_{h=1}^H \frac{S_h}{S_{\max}} = \lambda \cdot \text{sigmoid}(\text{span\_logits}).\text{mean}()
        """
        # Normalized span in [0, 1] per head
        norm_span = torch.sigmoid(self.span_logits)
        return lambda_span * norm_span.mean()

    def _relative_position_bucket(
        self, relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.relative_attention_bias is not None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)
            values = values.permute([2, 0, 1]).unsqueeze(0)
            return values
        return torch.zeros((1, self.num_heads, query_length, key_length), device=device)

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. HuggingFace / standard UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("AdaptiveSpanAttention UniversalAdapter fallback for %s: %s", proj, e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            c_attn_layer = src.c_attn
            base_c = getattr(c_attn_layer, "base_layer", c_attn_layer)
            weight = getattr(base_c, "weight", getattr(c_attn_layer, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_layer, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                hidden_dim = getattr(self, "hidden_size", 768)
                is_conv1d = (
                    c_attn_layer.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or hasattr(c_attn_layer, "nf")
                    or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[1] == 3 * hidden_dim)
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                q_w = weight[:split_size, :].contiguous()
                k_w = weight[split_size : 2 * split_size, :].contiguous()
                v_w = weight[2 * split_size :, :].contiguous()
                q_b = bias_data[:split_size].contiguous() if bias_data is not None else None
                k_b = bias_data[split_size : 2 * split_size].contiguous() if bias_data is not None else None
                v_b = bias_data[2 * split_size :].contiguous() if bias_data is not None else None
                extracted["q"] = (q_w, q_b)
                extracted["k"] = (k_w, k_b)
                extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_layer = src.c_proj
            base_o = getattr(c_proj_layer, "base_layer", c_proj_layer)
            w_o = getattr(base_o, "weight", getattr(c_proj_layer, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                hidden_dim = getattr(self, "hidden_size", 768)
                is_conv1d_proj = (
                    c_proj_layer.__class__.__name__ == "Conv1D"
                    or base_o.__class__.__name__ == "Conv1D"
                    or hasattr(c_proj_layer, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] != hidden_dim and w_o.shape[1] == hidden_dim)
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d_proj:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_layer, "bias", None))
                extracted["o"] = (w_o.contiguous(), b_o.data.clone().contiguous() if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
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

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        cfg_type = getattr(self.config, "model_type", "").lower() if self.config else ""
        if not self.is_nested_encoder:
            self.is_nested_encoder = bool(
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or cfg_type in ("roberta", "bert")
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower() and "gpt" not in cls_name.lower() and "qwen" not in cls_name.lower())
            )
        if self.is_nested_encoder and not isinstance(self.out_proj, IdentityOutProj):
            self.out_proj = IdentityOutProj()

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(f"AdaptiveSpanAttention missing required weights for projections: {missing}. Proceeding with initialized weights.")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"AdaptiveSpanAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if not self.is_nested_encoder and extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"AdaptiveSpanAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                if proj_key == "o" and self.is_nested_encoder:
                    continue
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w.shape
                    target_w.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w[:min(out_d, src_out), :min(in_d, src_in)])
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[:min(len(target.bias), len(b_val))].copy_(b_val[:min(len(target.bias), len(b_val))])

                target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # Apply RoPE (accept from position_embeddings or fallback to rotary_emb)
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(N_q, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(K, seq_len=N_k)
            except Exception:
                try:
                    cos, sin = rotary_mod(K, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        # GQA KV repetition
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale  # [B, H, N_q, N_k]

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            scores = apply_attention_mask(scores, mask)
        if position_bias is not None:
            scores = scores + position_bias.to(scores.dtype)

        # Explicit Causal Masking for autoregressive decoders
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )
        if is_decoder:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        # Softmax computation: guard against rows where all keys are masked (-inf / < -9000)
        all_masked = (scores.max(dim=-1, keepdim=True).values < -9000.0)

        if scores.dtype in (torch.float16, torch.bfloat16):
            c_sz = 256
            attn = torch.empty_like(scores)
            for s_idx in range(0, N_q, c_sz):
                e_idx = min(s_idx + c_sz, N_q)
                chunk_scores = scores[:, :, s_idx:e_idx, :].float()
                chunk_soft = F.softmax(chunk_scores, dim=-1)
                chunk_soft = torch.where(torch.isnan(chunk_soft), torch.zeros_like(chunk_soft), chunk_soft)
                attn[:, :, s_idx:e_idx, :] = chunk_soft.to(scores.dtype)
        else:
            raw_soft = F.softmax(scores.float(), dim=-1)
            raw_soft = torch.where(torch.isnan(raw_soft), torch.zeros_like(raw_soft), raw_soft)
            attn = raw_soft.to(scores.dtype)

        attn = attn.masked_fill(all_masked, 0.0)

        # Differentiable Adaptive Span Soft Ramp (Sukhbaatar et al., 2019)
        # Continuous piecewise-linear masking: g_z(d) = clamp((R + z - d) / R, 0.0, 1.0)
        # Avoids [H, N, N] boolean allocation and provides non-zero gradients to self.span_logits
        if not is_cross_attention and N_q == N_k:
            # Stronger fast-path: only allocate ramp when it can actually affect attention
            span = torch.sigmoid(self.span_logits) * self.max_span   # [H]
            # If per-head (span + ramp_size) >= N_k for all heads, ramp has no effect
            if not ((span + self.ramp_size) >= N_k).all():
                pos_q = torch.arange(N_q, device=device, dtype=scores.dtype).unsqueeze(1)  # [N_q, 1]
                pos_k = torch.arange(N_k, device=device, dtype=scores.dtype).unsqueeze(0)  # [1, N_k]
                if is_decoder:
                    dist = (pos_q - pos_k).clamp(min=0.0)  # [N_q, N_k]
                else:
                    dist = (pos_q - pos_k).abs()           # [N_q, N_k]

                span_h = span.view(1, H, 1, 1).to(scores.dtype)
                dist_mat = dist.view(1, 1, N_q, N_k)
                ramp = torch.clamp((self.ramp_size + span_h - dist_mat) / self.ramp_size, 0.0, 1.0)

                attn = attn * ramp
                attn_sum = attn.sum(dim=-1, keepdim=True)
                attn = torch.where(attn_sum > 1e-6, attn / attn_sum.clamp(min=1e-6), torch.zeros_like(attn))
                attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
                attn = attn.masked_fill(all_masked, 0.0)

        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if position_bias is not None:
            return output, position_bias
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adaptive_span_inject")
@ATTENTION_REGISTRY.register("adaptive_span")
@ATTENTION_REGISTRY.register("adaptive_span_attention")
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
    r"""Safe injection factory for AdaptiveSpanAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
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
        injected_blocks = []
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_span(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                injected_blocks.append(res)
        if injected_blocks:
            if hasattr(target, "_adaptive_span_modules"):
                target._adaptive_span_modules.extend(injected_blocks)
            else:
                target._adaptive_span_modules = injected_blocks
            return target
        return None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if hasattr(cand, "self"):
                    src_attn = cand.self
                else:
                    src_attn = cand
                break

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    # Skip if already injected
    if src_attn is not None:
        if isinstance(src_attn, AdaptiveSpanAttention):
            return None
        if "AdaptiveSpan" in src_attn.__class__.__name__:
            return None
        if hasattr(src_attn, "inner") and isinstance(getattr(src_attn, "inner"), AdaptiveSpanAttention):
            return None
        if hasattr(src_attn, "_wrapped_attention") and isinstance(getattr(src_attn, "_wrapped_attention"), AdaptiveSpanAttention):
            return None

    if target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            cand = getattr(target, attr, None)
            if cand is None:
                continue
            if isinstance(cand, AdaptiveSpanAttention):
                return None
            if "AdaptiveSpan" in cand.__class__.__name__:
                return None
            if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), AdaptiveSpanAttention):
                return None
            if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), AdaptiveSpanAttention):
                return None

    # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
    if src_attn is not None:
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

    hidden_size = (
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        getattr(src_attn, 'dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (src_attn.q_lin.in_features if hasattr(src_attn, 'q_lin') and hasattr(src_attn.q_lin, 'in_features') else None) or
        (getattr(target, 'in_features', None) if target is not None else None) or
        getattr(cfg, 'hidden_size', getattr(cfg, 'dim', 64))
    )
    num_heads = (
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(src_attn, 'n_heads', None) or
        getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_heads', 4))
    )

    rank = getattr(comp_config, "adaptive_span_rank", kwargs.get("rank", 32))
    dropout = getattr(comp_config, "dropout", kwargs.get("dropout", 0.1))
    max_span = getattr(comp_config, "max_span", kwargs.get("max_span", 128))

    model_type_str = str(getattr(cfg, "model_type", "")).lower() if cfg is not None else ""

    is_nested_layer = (
        (hasattr(target, "attention") and hasattr(target.attention, "output") and hasattr(target.attention.output, "dense"))
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
            **kwargs,
        )
    except Exception:
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