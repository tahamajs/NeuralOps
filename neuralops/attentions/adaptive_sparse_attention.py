"""
Adaptive Sparse Attention Module (Production-Hardened).
Implements dynamic top-k and threshold-based adaptive sparse attention with LoRA adapters,
multi-architecture weight extraction, RoPE rotary embeddings, causal/bidirectional masking,
and safe injection.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.model_family import ModelAdapter


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
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q = (
            torch.cat([q_rot_embed, q_pass], dim=-1)
            if q_pass.numel() > 0
            else q_rot_embed
        )
        k = (
            torch.cat([k_rot_embed, k_pass], dim=-1)
            if k_pass.numel() > 0
            else k_rot_embed
        )
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
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
        # dtype-safe: cast weight/bias to input dtype
        w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        b = None
        if self.bias is not None:
            b = self.bias if self.bias.dtype == x.dtype else self.bias.to(x.dtype)
        out = F.linear(x, w, b)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
            bb = (
                self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
            )
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register(
    "adaptive_sparse",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "adaptive_sparse_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "adaptive_sparse_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "adaptive_sparse_attention_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class AdaptiveSparseAttention(BaseAttention):
    is_custom_attention: bool = True
    supports_causal = True
    supports_bidirectional = True
    supports_gqa = True
    supports_rope = True
    triton_kernel: str = "sparse_attn"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        sparsity_ratio: float | None = None,
        dropout: float = 0.1,
        lora_rank: int = 16,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # Resolve hidden_size and num_heads safely even if config is None
        hidden_size = (
            kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
        )
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("n_heads")
            or kwargs.get("num_attention_heads")
        )
        num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")

        head_dim = 0
        if config is not None:
            adapter = ModelAdapter(config)
            hidden_size = hidden_size or adapter.hidden_size
            num_heads = num_heads or adapter.num_heads
            num_kv_heads = num_kv_heads or adapter.num_key_value_heads
            head_dim = getattr(adapter, "head_dim", 0)

        if not hidden_size and source_attn is not None:
            cand = getattr(
                source_attn,
                "hidden_size",
                getattr(
                    source_attn,
                    "embed_dim",
                    getattr(source_attn, "all_head_size", None),
                ),
            )
            if (
                cand is None
                and hasattr(source_attn, "q_proj")
                and hasattr(source_attn.q_proj, "in_features")
            ):
                cand = source_attn.q_proj.in_features
            hidden_size = cand

        hidden_size = int(hidden_size or 768)
        num_heads = int(num_heads or 12)
        num_kv_heads = int(num_kv_heads or num_heads)
        head_dim = (
            head_dim
            if head_dim > 0
            else (hidden_size // num_heads if num_heads > 0 else 64)
        )

        self.hidden_size: int = hidden_size
        self.d_model: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Decoder & Causal detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = (
            getattr(
                config,
                "is_decoder",
                getattr(config, "is_causal", kwargs.get("is_causal", False)),
            )
            if config
            else False
        )
        _mt = (model_type or "").lower()
        _decoder_prefixes = (
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
        if (
            any(_mt.startswith(p) for p in _decoder_prefixes)
            or hasattr(config, "architectures")
            and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            )
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)

        # Detect nested encoder architecture early
        source_cls = source_attn.__class__.__name__ if source_attn is not None else ""
        is_nested = source_cls in ("BertSelfAttention", "RobertaSelfAttention") or (
            source_cls.endswith(("SelfAttention", "SelfAttn"))
            and "distil" not in source_cls.lower()
        )
        has_out = (
            any(
                hasattr(source_attn, attr)
                for attr in (
                    "dense",
                    "out_proj",
                    "c_proj",
                    "out_lin",
                    "o_proj",
                    "wo",
                    "o",
                    "proj",
                )
            )
            if source_attn is not None
            else False
        )
        self.is_nested_encoder: bool = bool(
            kwargs.get("is_nested_encoder", (is_nested and not has_out))
        )

        # Head dim fallback from source_attn if head_dim <= 0 or not resolved properly
        if source_attn is not None and getattr(source_attn, "head_dim", 0):
            head_dim = source_attn.head_dim
            self.head_dim = head_dim
            self.all_head_size = self.num_heads * self.head_dim
            self.kv_head_size = self.num_kv_heads * self.head_dim
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Default sparsity: lower for bidirectional encoders (0.1) than decoders (0.5)
        # Check explicit parameter, kwargs, comp_config, config
        cfg_encoder_sparsity = getattr(
            comp_config,
            "adaptive_sparse_encoder_sparsity",
            kwargs.get("adaptive_sparse_encoder_sparsity"),
        )
        if cfg_encoder_sparsity is None and config is not None:
            cfg_encoder_sparsity = getattr(
                config, "adaptive_sparse_encoder_sparsity", None
            )

        if sparsity_ratio is not None:
            effective_sparsity = float(sparsity_ratio)
        elif "sparsity_ratio" in kwargs:
            effective_sparsity = float(kwargs["sparsity_ratio"])
        elif comp_config is not None and hasattr(
            comp_config, "adaptive_sparsity_ratio"
        ):
            effective_sparsity = float(comp_config.adaptive_sparsity_ratio)
        elif config is not None and hasattr(config, "adaptive_sparsity_ratio"):
            effective_sparsity = float(config.adaptive_sparsity_ratio)
        elif not self.is_decoder:
            if cfg_encoder_sparsity is not None:
                effective_sparsity = float(cfg_encoder_sparsity)
            else:
                effective_sparsity = 0.1
        else:
            effective_sparsity = 0.5

        self.default_sparsity: float = effective_sparsity

        # LoRA rank and alpha: independent reads
        if config is not None:
            self.lora_rank = getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
            self.lora_alpha = getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        elif comp_config is not None:
            self.lora_rank = getattr(
                comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)
            )
            self.lora_alpha = getattr(
                comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)
            )
        else:
            self.lora_rank = kwargs.get("lora_rank", lora_rank)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base: bool = self.lora_rank > 0

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
        out_lora_rank = 0 if self.is_nested_encoder else self.lora_rank
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=out_lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Aliases
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj

        # Per-head learnable sparsity logit initialized to target sparsity
        init_logit = math.log(
            max(1e-4, min(0.9999, effective_sparsity))
            / (1.0 - max(1e-4, min(0.9999, effective_sparsity)))
        )
        self.sparsity_logits = nn.Parameter(
            torch.full((self.num_heads,), fill_value=init_logit), requires_grad=True
        )
        self.output_scale = nn.Parameter(torch.ones(1))

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", dropout),
                ),
            )
            if config
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)
        self.source_attn = None
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None)
            if source_attn is not None
            else None
        )

        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: nn.Embedding | None = None
        if config is not None:
            self.relative_attention_num_buckets: int = getattr(
                config, "relative_attention_num_buckets", 32
            )
            self.relative_attention_max_distance: int = getattr(
                config, "relative_attention_max_distance", 128
            )
        else:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128

        self.triton_kwargs = {"block_size": 64, "eps": 1e-6}

        if source_attn is not None:
            self.copy_weights(source_attn)

    def freeze_base_weights(self) -> None:
        """Freeze base projection weights when LoRA adapters are present."""
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection weights."""
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(
        self, query_length: int, key_length: int, device: torch.device | None = None
    ) -> torch.Tensor:
        if self.relative_attention_bias is not None:
            context_position = torch.arange(
                query_length, dtype=torch.long, device=device
            )[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
                None, :
            ]
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
        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
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
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn / c_proj (handling Conv1D and LoRALinear wrapper)
        if (
            hasattr(src, "c_attn")
            and src.c_attn is not None
            and (
                extracted["q"][0] is None
                or extracted["k"][0] is None
                or extracted["v"][0] is None
            )
        ):
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or (weight.ndim == 2 and weight.shape[0] < weight.shape[1])
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    (
                        bias_data[split_size : 2 * split_size]
                        if bias_data is not None
                        else None
                    ),
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if (
            hasattr(src, "c_proj")
            and src.c_proj is not None
            and extracted["o"][0] is None
        ):
            c_proj_mod = src.c_proj
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                if (
                    (c_proj_mod.__class__.__name__ == "Conv1D")
                    or (base_o.__class__.__name__ == "Conv1D")
                    or (w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (
                    w_o,
                    bias_o.data.clone() if bias_o is not None else None,
                )

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = src.output.dense
            extracted["o"] = (
                dense.weight.data.clone(),
                dense.bias.data.clone() if dense.bias is not None else None,
            )

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
                            extracted[proj] = (
                                layer.weight.data.clone(),
                                getattr(layer, "bias", None),
                            )
                            break

        # Detect nested encoder architecture
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
            or (hasattr(src, "query") and not has_out_proj)
            or (hasattr(src, "self") and hasattr(src, "output"))
        ) and not has_out_proj

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AdaptiveSparseAttention missing required weights for projections: {missing}"
            )

        # Frobenius norm validation on slices
        for proj_key, target_mod in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
        ]:
            w = extracted[proj_key][0]
            if w is not None:
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                norm = float(torch.norm(w[:out_d, :in_d].float()).item())
                if norm < 1e-4:
                    raise RuntimeError(
                        f"AdaptiveSparseAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm:.6f})"
                    )

        if extracted["o"][0] is not None:
            w_o = extracted["o"][0]
            out_d = min(self.out_proj.weight.size(0), w_o.size(0))
            in_d = min(self.out_proj.weight.size(1), w_o.size(1))
            norm_o = float(torch.norm(w_o[:out_d, :in_d].float()).item())
            if norm_o < 1e-4:
                raise RuntimeError(
                    f"AdaptiveSparseAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o:.6f})"
                )

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w.shape
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w[: min(out_d, src_out), : min(in_d, src_in)]
                    )
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                            b_val[: min(len(target.bias), len(b_val))]
                        )
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if (
            hasattr(src, "relative_attention_bias")
            and src.relative_attention_bias is not None
        ):
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    def get_sparsity_loss(self) -> torch.Tensor:
        r"""Regularization loss encouraging head sparsity."""
        return torch.sigmoid(self.sparsity_logits).mean()

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, N_q, H, d).transpose(1, 2).contiguous()
        K = (
            self.k_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(kv_input)
            .view(B, N_k, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        # Apply RoPE (accept from position_embeddings or fallback to rotary_emb)
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if (
            pos_emb is not None
            and isinstance(pos_emb, (tuple, list))
            and len(pos_emb) >= 2
        ):
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(N_q, device=device).unsqueeze(0).expand(B, -1)
            # 1. Try rotary_module(Q, position_ids)
            # 2. Try rotary_module(Q, seq_len=...)
            # 3. Try rotary_module(K, ...)
            cos, sin = None, None
            for probe_t in (Q, K):
                try:
                    res = rotary_mod(probe_t, pos_ids)
                    if isinstance(res, (tuple, list)) and len(res) >= 2:
                        cos, sin = res[0], res[1]
                        break
                except Exception as e:
                    logger.warning("RoPE embedding failed on (probe_t, pos_ids): %s", e)
                try:
                    res = rotary_mod(probe_t, seq_len=N_k)
                    if isinstance(res, (tuple, list)) and len(res) >= 2:
                        cos, sin = res[0], res[1]
                        break
                except Exception as e:
                    logger.warning("RoPE embedding failed on (probe_t, seq_len): %s", e)
                try:
                    res = rotary_mod(probe_t)
                    if isinstance(res, (tuple, list)) and len(res) >= 2:
                        cos, sin = res[0], res[1]
                        break
                except Exception as e:
                    logger.warning("RoPE embedding failed on (probe_t): %s", e)
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        chunk_size = 64
        scale = d**-0.5

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias")
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")

        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "")
            in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )

        sparsity_per_head = torch.sigmoid(self.sparsity_logits).clamp(0.0, 0.95)  # [H]
        k_per_head = torch.clamp(
            ((1.0 - sparsity_per_head) * N_k).long(), min=1, max=N_k
        )  # [H]
        k_indices_base = (k_per_head - 1).view(1, H, 1, 1)

        valid_m = None
        if mask is not None and mask.dim() != 4:
            valid_m = get_token_valid_mask(mask, B, N_k, device)
            if valid_m is None:
                valid_m = torch.ones(B, N_k, device=device, dtype=torch.bool)

        context_chunks = []
        attn_chunks = [] if output_attentions else None

        for q_start in range(0, N_q, chunk_size):
            q_end = min(q_start + chunk_size, N_q)
            q_chunk = Q[:, :, q_start:q_end, :]  # [B, H, C_q, d]
            C_q = q_end - q_start

            scores_chunk = (
                torch.matmul(q_chunk, K.transpose(-2, -1).contiguous()) * scale
            )  # [B, H, C_q, N_k]

            if position_bias is not None:
                pb_chunk = position_bias[:, :, q_start:q_end, :]
                if mask is not None and mask.dim() == 4:
                    pb_chunk = pb_chunk + mask[:, :, q_start:q_end, :].to(
                        scores_chunk.dtype
                    )
                scores_chunk = scores_chunk + pb_chunk.to(scores_chunk.dtype)
            elif mask is not None:
                if mask.dim() == 4:
                    scores_chunk = scores_chunk + mask[:, :, q_start:q_end, :].to(
                        scores_chunk.dtype
                    )
                elif valid_m is not None:
                    scores_chunk = scores_chunk.masked_fill(
                        (~valid_m).unsqueeze(1).unsqueeze(2), -10000.0
                    )

            if is_decoder:
                q_idx = torch.arange(q_start, q_end, device=device).view(1, 1, C_q, 1)
                k_idx = torch.arange(N_k, device=device).view(1, 1, 1, N_k)
                causal_chunk = k_idx <= q_idx
                scores_chunk = scores_chunk.masked_fill(~causal_chunk, -10000.0)

            # Vectorized Top-K Thresholding on chunk with differentiable soft sigmoid threshold
            if not is_cross_attention:
                sorted_scores, _ = torch.sort(scores_chunk, dim=-1, descending=True)
                k_indices = k_indices_base.expand(B, H, C_q, 1)
                topk_thresh = torch.gather(sorted_scores, dim=-1, index=k_indices)
                temperature = 0.1
                soft_mask = torch.sigmoid((scores_chunk - topk_thresh) / temperature)
                scores_chunk = scores_chunk + torch.log(soft_mask + 1e-8)

            attn_chunk = torch.nan_to_num(
                F.softmax(scores_chunk.float(), dim=-1), nan=0.0
            ).to(dtype)
            attn_chunk = self.dropout(attn_chunk)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    hm = head_mask.unsqueeze(1)
                elif head_mask.dim() == 4:
                    hm = head_mask[:, :, q_start:q_end, :]
                else:
                    hm = head_mask
                attn_chunk = attn_chunk * hm

            ctx_chunk = torch.matmul(attn_chunk, V)  # [B, H, C_q, d]
            context_chunks.append(ctx_chunk)
            if output_attentions:
                attn_chunks.append(attn_chunk)

        context = torch.cat(context_chunks, dim=2)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        attn = torch.cat(attn_chunks, dim=2) if output_attentions else None

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if position_bias is not None:
            return output * self.output_scale, position_bias
        return output * self.output_scale, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adaptive_sparse_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_sparse_inject")
@ATTENTION_REGISTRY.register("adaptive_sparse_attn_inject")
@ATTENTION_REGISTRY.register("inject_adaptive_sparse")
@ATTENTION_REGISTRY.register("inject_adaptive_sparse_attention")
def inject_adaptive_sparse(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> AdaptiveSparseAttention | nn.Module | None:
    r"""Safe injection factory for AdaptiveSparseAttention across diverse transformer architectures."""
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_sparse(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    sparsity_ratio = getattr(
        comp_config, "adaptive_sparsity_ratio", kwargs.get("sparsity_ratio")
    )
    dropout = getattr(comp_config, "dropout", kwargs.get("dropout", 0.1))

    try:
        new_block = AdaptiveSparseAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            sparsity_ratio=sparsity_ratio,
            dropout=dropout,
            pretrained_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"inject_adaptive_sparse failed: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
                # Outer dense wrapping for nested encoder
                if hasattr(target.attention, "output") and hasattr(
                    target.attention.output, "dense"
                ):
                    outer_dense = target.attention.output.dense
                    if not isinstance(outer_dense, LoRALinear):
                        wrapped_dense = LoRALinear(
                            outer_dense.in_features,
                            outer_dense.out_features,
                            bias=(outer_dense.bias is not None),
                            lora_rank=new_block.lora_rank,
                            lora_alpha=new_block.lora_alpha,
                            freeze_base=True,
                        )
                        wrapped_dense.weight.data.copy_(outer_dense.weight.data)
                        if (
                            outer_dense.bias is not None
                            and wrapped_dense.bias is not None
                        ):
                            wrapped_dense.bias.data.copy_(outer_dense.bias.data)
                        target.attention.output.dense = wrapped_dense
            else:
                target.attention = new_block
        elif hasattr(target, "self"):
            target.self = new_block
        elif hasattr(target, "attn"):
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block

    return new_block
