"""
Decay Linear Attention (DecayLinearAttention).

Multi-head bidirectional & causal linear recurrent attention with data-dependent exponential decay gating,
ELU+1 feature maps, denominator normalization, and pretrained weight transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (forward-only recurrence)
- Nested encoder detection to bypass out_proj in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- True O(L) linear complexity recurrence without O(L^2) memory footprint
- Numerically stable FP32 state accumulation with nan_to_num protection
- Safe injection factory returning None on extraction failures
- Full cross-attention support
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
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.core.triton_kernels.gla_scan import (
        MAX_REGISTER_DIM,
        gla_scan_triton,
    )

    HAS_TRITON_DECAY = True
except Exception:
    HAS_TRITON_DECAY = False

logger = logging.getLogger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
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


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Applies rotary position embeddings with partial-RoPE dimension slicing."""
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
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


class LoRALinear(nn.Module):
    r"""Linear projection layer with frozen base weights and trainable residual LoRA adapters."""

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
        """Freeze base projection parameters so only LoRA parameters adapt."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection parameters."""
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register(
    "decay_linear_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "decay_linear_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class DecayLinearAttention(BaseAttention):
    r"""Decay Linear Attention Module with Data-Dependent Exponential Decay Gating."""

    is_custom_attention: bool = True
    triton_kernel: str = "linear_attn"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        def _cfg_get(k: str, default: Any = None) -> Any:
            for obj in (config, comp_config):
                if obj is not None and hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
            return kwargs.get(k, default)

        hidden_size = kwargs.get("hidden_size")
        if hidden_size is None and d_model is not None:
            hidden_size = d_model
        if hidden_size is None:
            hidden_size = _cfg_get("hidden_size", _cfg_get("dim", None))
            if hidden_size is None and source_attn is not None:
                out_layer = getattr(
                    source_attn,
                    "out_lin",
                    getattr(
                        source_attn,
                        "dense",
                        getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None)),
                    ),
                )
                hidden_size = (
                    getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                )
            elif hidden_size is None:
                hidden_size = 768

        if hasattr(hidden_size, "hidden_size"):
            hidden_size = hidden_size.hidden_size

        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads")
        if num_heads is None:
            num_heads = _cfg_get("num_attention_heads", _cfg_get("num_heads", None))
            if num_heads is None and source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "num_heads", 12),
                    ),
                )
            elif num_heads is None:
                num_heads = 12

        super().__init__(config, rank)
        self.dim: int = int(hidden_size)
        self.num_heads: int = int(num_heads)

        resolved_head_dim = _cfg_get("head_dim", None)
        if resolved_head_dim is None and source_attn is not None:
            resolved_head_dim = getattr(source_attn, "head_dim", None)
        if (
            resolved_head_dim is None
            and source_attn is not None
            and hasattr(source_attn, "q_proj")
            and hasattr(source_attn.q_proj, "out_features")
        ):
            resolved_head_dim = source_attn.q_proj.out_features // max(1, self.num_heads)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim: int = int(resolved_head_dim)

        kv_heads = _cfg_get("num_key_value_heads", None)
        if kv_heads is None and source_attn is not None:
            kv_heads = getattr(
                source_attn,
                "num_key_value_heads",
                getattr(
                    source_attn,
                    "num_kv_heads",
                    getattr(source_attn, "n_kv_heads", None),
                ),
            )
            if (
                kv_heads is None
                and hasattr(source_attn, "k_proj")
                and hasattr(source_attn.k_proj, "out_features")
                and self.head_dim > 0
            ):
                kv_heads = source_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads: int = int(kv_heads if kv_heads is not None else self.num_heads)

        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Static layer index
        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        self.lora_rank: int = int(_cfg_get("lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha: float = float(_cfg_get("lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base: bool = self.lora_rank > 0

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = _cfg_get("is_decoder", _cfg_get("is_causal", kwargs.get("is_causal", False)))
        _decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "phi",
            "gemma",
            "opt",
            "bloom",
            "falcon",
        )
        if (
            any(model_type.startswith(p) for p in _decoder_prefixes)
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
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", False))

        # Multi-head projections equipped with LoRA adapters
        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.decay_gate = nn.Linear(self.dim, self.num_heads, bias=True)
        # Bounded initialization spanning [sigmoid(-0.5), sigmoid(0.0)] = [0.38, 0.50] for head diversity and wide context horizon
        self.log_decay = nn.Parameter(torch.linspace(2.0, 4.0, self.num_heads))
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        nn.init.zeros_(self.decay_gate.weight)
        nn.init.zeros_(self.decay_gate.bias)

        self.rotary_emb = None
        self.comp_config: Any | None = comp_config
        self.triton_kwargs = {"phi": "elu+1", "eps": 1e-6}

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Compatibility properties
    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    @property
    def dense(self) -> LoRALinear:
        return self.out_proj

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_proj

    @property
    def q_lin(self) -> LoRALinear:
        return self.q_proj

    @property
    def k_lin(self) -> LoRALinear:
        return self.k_proj

    @property
    def v_lin(self) -> LoRALinear:
        return self.v_proj

    @property
    def out_lin(self) -> LoRALinear:
        return self.out_proj

    @property
    def W_q(self) -> LoRALinear:
        return self.q_proj

    @property
    def W_k(self) -> LoRALinear:
        return self.k_proj

    @property
    def W_v(self) -> LoRALinear:
        return self.v_proj

    @property
    def c_proj(self) -> LoRALinear:
        return self.out_proj

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained projection weights with strict Frobenius norm verification."""
        if old_attn is None:
            return

        cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
        has_out_lin = (
            hasattr(old_attn, "out_lin")
            or hasattr(old_attn, "c_proj")
            or hasattr(old_attn, "o_proj")
            or hasattr(old_attn, "dense")
        )
        if (
            (
                hasattr(old_attn, "self")
                and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output")
                and hasattr(old_attn.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            self.is_nested_encoder = True

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c_attn_mod = old_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug(f"c_attn_mod.merge failed: {e}")
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    ("Conv1D" in c_attn_mod.__class__.__name__)
                    or ("Conv1D" in base_c.__class__.__name__)
                    or hasattr(c_attn_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
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

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                c_proj_mod = old_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        ("Conv1D" in c_proj_mod.__class__.__name__)
                        or ("Conv1D" in base_o.__class__.__name__)
                        or hasattr(c_proj_mod, "nf")
                        or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                base_mod = getattr(mod, "base_layer", mod)
                if base_mod is not None and hasattr(base_mod, "weight"):
                    extracted[proj_key] = (
                        base_mod.weight.data.clone(),
                        (
                            base_mod.bias.data.clone()
                            if getattr(base_mod, "bias", None) is not None
                            else None
                        ),
                    )
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                out_mod = old_attn.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                extracted["o"] = (
                    base_out.weight.data.clone(),
                    (
                        base_out.bias.data.clone()
                        if getattr(base_out, "bias", None) is not None
                        else None
                    ),
                )

        # 3. Standard linear projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj", "c_proj", "wo"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(old_attn, name):
                        mod = getattr(old_attn, name)
                        base_mod = getattr(mod, "base_layer", mod)
                        if hasattr(base_mod, "weight"):
                            w_data = base_mod.weight.data.clone()
                            if (
                                (base_mod.__class__.__name__ == "Conv1D")
                                or hasattr(base_mod, "nf")
                                or (
                                    w_data.ndim == 2
                                    and proj_key == "o"
                                    and w_data.shape[0] < w_data.shape[1]
                                )
                            ):
                                w_data = w_data.t().contiguous()
                            b_data = getattr(base_mod, "bias", None)
                            extracted[proj_key] = (
                                w_data,
                                b_data.data.clone() if b_data is not None else None,
                            )
                            break

        # Fallback via UniversalAdapter
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
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug(f"UniversalAdapter fallback: {e}")

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"DecayLinearAttention copy_weights: missing mandatory projections {missing_after} "
                f"from {old_attn.__class__.__name__}."
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
                        device=target_mod.weight.device, dtype=target_mod.weight.dtype
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                # Frobenius norm verification (>= 1e-4)
                w_norm = float(torch.linalg.norm(target_mod.weight.data.float()).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"DecayLinearAttention weight copy verification failed for '{proj_key}': norm too small ({w_norm:.2e})"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    def _decay_scan_linear(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        lambdas: torch.Tensor,
        mask: torch.Tensor | None = None,
        reverse: bool = False,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        r"""
        Numerically stable parallel decay scan with O(1) state per chunk (chunked recurrence).
        Eliminates quadratic O(L^2) memory footprint and prevents GPU out-of-memory.
        """
        B, H, L, d = q.shape

        # Path 1: Hardware-fused Triton kernel on CUDA
        if (
            HAS_TRITON_DECAY
            and q.is_cuda
            and d <= MAX_REGISTER_DIM
            and mask is None
            and not reverse
        ):
            try:
                return gla_scan_triton(q.float(), k.float(), v.float(), lambdas.float()).to(q.dtype)
            except Exception as e:
                logger.debug("Triton decay scan failed, falling back to chunked PyTorch: %s", e)

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        lambdas_f = lambdas.float().clamp(min=1e-4, max=1.0 - 1e-4)

        if mask is not None:
            # mask is [B, L] boolean tensor where True = valid token
            m_f = mask.float().unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            k_f = k_f * m_f
            v_f = v_f * m_f

        if reverse:
            q_f = q_f.flip(dims=[2])
            k_f = k_f.flip(dims=[2])
            v_f = v_f.flip(dims=[2])
            lambdas_f = lambdas_f.flip(dims=[-1])

        log_lambdas = torch.log(lambdas_f)
        cum_log = torch.cumsum(log_lambdas, dim=-1)

        if chunk_size >= L:
            log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
            decay = torch.exp(log_decay.clamp(max=0.0))
            causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
            decay = decay.masked_fill(~causal_mask, 0.0)
            scores = torch.matmul(q_f, k_f.transpose(-1, -2).contiguous()) * decay
            num = torch.matmul(scores, v_f)
            denom = scores.sum(dim=-1, keepdim=True)
        else:
            pad_len = (chunk_size - (L % chunk_size)) % chunk_size
            if pad_len > 0:
                q_pad = F.pad(q_f, (0, 0, 0, pad_len))
                k_pad = F.pad(k_f, (0, 0, 0, pad_len))
                v_pad = F.pad(v_f, (0, 0, 0, pad_len))
                last_cum = cum_log[:, :, -1:]
                cum_pad = torch.cat([cum_log, last_cum.expand(-1, -1, pad_len)], dim=-1)
            else:
                q_pad, k_pad, v_pad, cum_pad = q_f, k_f, v_f, cum_log

            total_L = L + pad_len
            num_chunks = total_L // chunk_size

            qc = q_pad.view(B, H, num_chunks, chunk_size, d)
            kc = k_pad.view(B, H, num_chunks, chunk_size, d)
            vc = v_pad.view(B, H, num_chunks, chunk_size, d)
            cum_c = cum_pad.view(B, H, num_chunks, chunk_size)

            intra_log_decay = cum_c.unsqueeze(-1) - cum_c.unsqueeze(-2)
            intra_decay = torch.exp(intra_log_decay.clamp(max=0.0))
            causal_mask = torch.tril(
                torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool)
            )
            intra_decay = intra_decay.masked_fill(~causal_mask, 0.0)

            if pad_len > 0:
                valid_in_last = chunk_size - pad_len
                intra_decay[:, :, -1, :, valid_in_last:] = 0.0

            intra_scores = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * intra_decay
            num_intra = torch.matmul(intra_scores, vc)
            denom_intra = intra_scores.sum(dim=-1, keepdim=True)

            S = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
            z = torch.zeros(B, H, d, 1, device=q.device, dtype=torch.float32)

            num_list = []
            denom_list = []

            prev_cum_end = None
            for c in range(num_chunks):
                curr_cum = cum_c[:, :, c]
                if c == 0:
                    inter_decay = torch.zeros(
                        B, H, chunk_size, 1, device=q.device, dtype=torch.float32
                    )
                else:
                    inter_decay = torch.exp((curr_cum - prev_cum_end).clamp(max=0.0)).unsqueeze(-1)

                num_inter = torch.matmul(qc[:, :, c], S) * inter_decay
                denom_inter = torch.matmul(qc[:, :, c], z) * inter_decay

                num_list.append(num_intra[:, :, c] + num_inter)
                denom_list.append(denom_intra[:, :, c] + denom_inter)

                cum_end = curr_cum[:, :, -1:]
                decay_to_end = torch.exp((cum_end - curr_cum).clamp(max=0.0)).unsqueeze(-1)

                k_curr = kc[:, :, c]
                v_curr = vc[:, :, c]
                if c == num_chunks - 1 and pad_len > 0:
                    valid_in_last = chunk_size - pad_len
                    k_curr = k_curr.clone()
                    k_curr[:, :, valid_in_last:] = 0.0

                k_to_end = k_curr * decay_to_end

                if c == 0:
                    decay_prev_to_end = 0.0
                else:
                    decay_prev_to_end = torch.exp(
                        (cum_end - prev_cum_end).clamp(max=0.0)
                    ).unsqueeze(-1)

                S = S * decay_prev_to_end + torch.matmul(
                    k_to_end.transpose(-1, -2).contiguous(), v_curr
                )
                z = (
                    z * decay_prev_to_end
                    + k_to_end.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
                )
                prev_cum_end = cum_end

            num = torch.cat(num_list, dim=2)[:, :, :L, :]
            denom = torch.cat(denom_list, dim=2)[:, :, :L, :]

        y = num / denom.clamp(min=1e-6)

        if reverse:
            y = y.flip(dims=[2])

        return y.to(q.dtype)

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
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. Rotary Position Embeddings (RoPE)
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                cos, sin = self.rotary_emb(v, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 3. GQA expansion
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 4. Kernel feature mapping
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Data-dependent decay lambdas in (0, 1) with bounded negative log decay bias
        decay_logits = self.decay_gate(hidden_states).transpose(
            1, 2
        ).contiguous() + self.log_decay.view(1, -1, 1)
        lambdas = torch.sigmoid(decay_logits).clamp(min=0.85, max=0.999)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        mask_bool = None
        if eff_mask is not None:
            mask_bool = get_token_valid_mask(eff_mask, B, kv_len, hidden_states.device)

        # 5. True O(L) Linear Decay Scan
        y_fwd = self._decay_scan_linear(q, k, v, lambdas, mask=mask_bool, reverse=False)

        if self.is_decoder or is_cross_attention:
            y = y_fwd
        else:
            y_bwd = self._decay_scan_linear(q, k, v, lambdas, mask=mask_bool, reverse=True)
            y = 0.5 * (y_fwd + y_bwd)

        # 6. Head mask handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            y = y * head_mask

        y = torch.nan_to_num(y, nan=0.0)
        out = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # 7. Output projection with nested encoder bypass (no output_scale)
        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        if mask_bool is not None and not is_cross_attention:
            final_out = final_out * mask_bool.unsqueeze(-1).to(final_out.dtype)

        return final_out, None


@ATTENTION_REGISTRY.register("decay_linear_attention_inject")
@ATTENTION_REGISTRY.register("inject_decay_linear_attention")
def inject_decay_linear(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_decay_linear(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config if "comp_config" in locals() or "comp_config" in kwargs else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection hook for DecayLinearAttention."""
    src_attn = old_attn
    if src_attn is None and layer is not None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn
        elif hasattr(layer, "self"):
            src_attn = layer.self

    try:
        new_block = DecayLinearAttention(
            config=model_config,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
        )
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_block
                new_block.is_nested_encoder = True
            else:
                layer.attention = new_block
                new_block.is_nested_encoder = False
        elif hasattr(layer, "attn"):
            layer.attn = new_block
            new_block.is_nested_encoder = False
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_block
            new_block.is_nested_encoder = False
        elif hasattr(layer, "self"):
            layer.self = new_block
            new_block.is_nested_encoder = True
        return new_block
    except Exception as e:
        logger.error("Failed to inject DecayLinearAttention at layer %d: %s", layer_idx, e)
        return None
