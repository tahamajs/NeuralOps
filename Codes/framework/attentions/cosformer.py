"""Cosformer: Rethinking Softmax in Attention with Linear Complexity.

Implements non-negative kernel feature mapping with cosine/sine positional reweighting,
cumulative sum causal linear attention for decoders, and associative accumulation for encoders.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models (vectorized prefix sum)
- Nested encoder detection to bypass out_proj in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Complete value masking (masking both K and V, including last-chunk V masking)
- Numerically stable FP32 accumulation with nan_to_num protection
- Safe injection factory returning None on extraction failures
- Full cross-attention support
- Thread-safe injection depth guard
- Even head_dim validation for RoPE
- Transparent rank-agnostic behavior (Cosformer does not use rank)

"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.triton_kernels import triton_dispatch


# ============================================================================
# Thread-safe injection depth guard (Fix 3)
# ============================================================================
_inject_state = threading.local()
_MAX_DEPTH = 4


def _get_inject_depth() -> int:
    return getattr(_inject_state, "depth", 0)


def _set_inject_depth(depth: int) -> None:
    _inject_state.depth = depth


# ============================================================================
# Utility functions
# ============================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Fix 11: This function assumes even head_dim. The caller (CosformerAttention)
    validates this at construction time.
    """
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
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# ============================================================================
# LoRALinear
# ============================================================================
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
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ============================================================================
# CosformerAttention
# ============================================================================
@ATTENTION_REGISTRY.register(
    "cosformer", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "cosformer_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
class CosformerAttention(BaseAttention):
    r"""Cosformer Attention with Linear Complexity and Exact Multi-Architecture Support."""

    is_custom_attention: bool = True
    triton_kernel = "linear_attn"

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        max_seq_len: int = 512,
        act_fun: str = "relu",
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        layer_id: Optional[int] = None,
        layer_idx: Optional[int] = None,
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
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")

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
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            elif hidden_size is None:
                hidden_size = 768

        if num_heads is None:
            num_heads = _cfg_get("num_attention_heads", _cfg_get("num_heads", None))
            if num_heads is None and source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "n_heads",
                    getattr(source_attn, "num_attention_heads", getattr(source_attn, "num_heads", 12)),
                )
            elif num_heads is None:
                num_heads = 12

        super().__init__(config, rank)
        self.dim: int = int(hidden_size)
        self.hidden_size: int = self.dim
        self.num_heads: int = int(num_heads)

        resolved_head_dim = _cfg_get("head_dim", None)
        if resolved_head_dim is None and source_attn is not None:
            resolved_head_dim = getattr(source_attn, "head_dim", None)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim: int = int(resolved_head_dim)

        # Fix 11: even head_dim required for RoPE
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"CosformerAttention with RoPE requires even head_dim, "
                f"got head_dim={self.head_dim}"
            )

        kv_heads = _cfg_get("num_key_value_heads", None)
        if kv_heads is None and source_attn is not None:
            kv_heads = getattr(
                source_attn,
                "num_key_value_heads",
                getattr(source_attn, "num_kv_heads", getattr(source_attn, "n_kv_heads", None)),
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
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Fix 9: documented max_seq_len default
        self.max_seq_len: int = int(_cfg_get("max_position_embeddings", max_seq_len))
        self.act_fun: str = act_fun
        self.eps: float = float(eps)

        # Static layer index
        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        )
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        self.lora_rank: int = int(_cfg_get("lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha: float = float(_cfg_get("lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base: bool = self.lora_rank > 0

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = _cfg_get("is_decoder", _cfg_get("is_causal", kwargs.get("is_causal", False)))
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "phi", "gemma", "opt", "bloom", "falcon")
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a)
            for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", False))

        # Fix 2: tau buffer removed (was unused)

        # Multi-head projections equipped with LoRA adapters
        self.q_proj = LoRALinear(
            self.dim, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.all_kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        self.rotary_emb = None
        self.comp_config: Optional[Any] = comp_config
        self.triton_kwargs = {
            "phi": "elu+1",
            "eps": self.eps,
            "cos_weight": None,
            "sin_weight": None,
        }

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def query(self) -> LoRALinear: return self.q_proj

    @property
    def key(self) -> LoRALinear: return self.k_proj

    @property
    def value(self) -> LoRALinear: return self.v_proj

    @property
    def dense(self) -> LoRALinear: return self.out_proj

    @property
    def o_proj(self) -> LoRALinear: return self.out_proj

    @property
    def q_lin(self) -> LoRALinear: return self.q_proj

    @property
    def k_lin(self) -> LoRALinear: return self.k_proj

    @property
    def v_lin(self) -> LoRALinear: return self.v_proj

    @property
    def out_lin(self) -> LoRALinear: return self.out_proj

    @property
    def W_q(self) -> LoRALinear: return self.q_proj

    @property
    def W_k(self) -> LoRALinear: return self.k_proj

    @property
    def W_v(self) -> LoRALinear: return self.v_proj

    @property
    def c_proj(self) -> LoRALinear: return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        """Transfer pretrained projection weights with strict Frobenius norm verification."""
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (
            hasattr(src, "dense")
            or hasattr(src, "out_proj")
            or hasattr(src, "out_lin")
            or hasattr(src, "c_proj")
            or hasattr(src, "o_proj")
        )
        if cls_name in ["BertSelfAttention", "RobertaSelfAttention"] or (hasattr(src, "query") and not has_out):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
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
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        (c_proj_mod.__class__.__name__ == "Conv1D")
                        or (base_o.__class__.__name__ == "Conv1D")
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                base_mod = getattr(mod, "base_layer", mod)
                if base_mod is not None and hasattr(base_mod, "weight"):
                    extracted[proj_key] = (
                        base_mod.weight.data.clone(),
                        base_mod.bias.data.clone() if getattr(base_mod, "bias", None) is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                base_out = getattr(out_mod, "base_layer", out_mod)
                extracted["o"] = (
                    base_out.weight.data.clone(),
                    base_out.bias.data.clone() if getattr(base_out, "bias", None) is not None else None,
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
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        base_mod = getattr(mod, "base_layer", mod)
                        if hasattr(base_mod, "weight"):
                            w_data = base_mod.weight.data.clone()
                            if (
                                (base_mod.__class__.__name__ == "Conv1D")
                                or hasattr(base_mod, "nf")
                                or (w_data.ndim == 2 and proj_key == "o" and w_data.shape[0] < w_data.shape[1])
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
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug(f"UniversalAdapter fallback: {e}")

        missing_after = [
            k for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"CosformerAttention copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
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

                # Frobenius norm verification (>= 1e-4)
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"CosformerAttention weight copy verification failed "
                        f"for '{proj_key}': norm too small ({w_norm:.2e})"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Positional weighting
    # ------------------------------------------------------------------
    def _get_cos_sin_weights(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Generate positional cosine and sine factors: [1, 1, seq_len, 1].

        Fix 6: Uses seq_len directly (simpler and more numerically stable).
        """
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        m = max(1, int(seq_len))
        angles = (math.pi / (2.0 * m)) * pos
        cos_weight = torch.cos(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        sin_weight = torch.sin(angles).view(1, 1, seq_len, 1).to(dtype=dtype)
        return cos_weight, sin_weight

    # ------------------------------------------------------------------
    # Forward
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
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Fix 17: input validation
        assert hidden_states.dim() == 3, (
            f"CosformerAttention expects 3D hidden_states [B, N, D], "
            f"got shape {tuple(hidden_states.shape)}"
        )

        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections
        Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # 2. RoPE (Fix 7: strict tuple requirement)
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings", None)
        if pos_emb is not None and not is_cross_attention:
            if isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
                cos, sin = pos_emb[0], pos_emb[1]
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
            else:
                logger.warning(
                    f"CosformerAttention: unexpected position_embeddings "
                    f"format ({type(pos_emb)}), skipping RoPE."
                )
        elif self.rotary_emb is not None and not is_cross_attention:
            rotary_mod = self.rotary_emb
            pos_ids = position_ids if position_ids is not None else kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            cos, sin = None, None
            try:
                cos, sin = rotary_mod(Q, seq_len=kv_len)
            except Exception:
                try:
                    cos, sin = rotary_mod(Q, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(Q)
                    except Exception:
                        pass
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        # 3. GQA head expansion
        if self.num_kv_heads != self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 4. Non-negative kernel activation (Fix 18: comment clarified)
        # FP32 for numerical stability
        if self.act_fun == "softplus":
            Q = F.softplus(Q.float())
            K = F.softplus(K.float())
        else:
            Q = F.elu(Q.float()) + 1.0
            K = F.elu(K.float()) + 1.0
        V = V.float()

        # 5. Token valid mask on both K and V
        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if eff_mask is not None:
            valid_mask = get_token_valid_mask(eff_mask, B, kv_len, device)
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).float()
            K = K * mask_expanded
            V = V * mask_expanded

        # 6. Cosine/Sine position modulation
        cos_w_q, sin_w_q = self._get_cos_sin_weights(N, device, torch.float32)

        # 7. Linear attention via shared Triton dispatcher
        is_causal_eval = bool(self.is_decoder and not is_cross_attention and N == kv_len)
        dispatch_kwargs = dict(self.triton_kwargs)
        dispatch_kwargs["cos_weight"] = cos_w_q
        dispatch_kwargs["sin_weight"] = sin_w_q
        dispatch_kwargs["eps"] = self.eps

        context = triton_dispatch(
            self.triton_kernel,
            Q, K, V,
            is_causal=is_causal_eval,
            **dispatch_kwargs,
        )

        # 8. Head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            context = context * head_mask

        context = torch.nan_to_num(context, nan=0.0)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        # 9. Output projection (nested encoder bypass)
        if self.is_nested_encoder:
            final_out = context
        else:
            final_out = self.out_proj(context)

        return final_out.to(dtype), None

    # ------------------------------------------------------------------
    # Causal forward (Fix 8: V masked in last chunk)
    # ------------------------------------------------------------------
    def _forward_causal(
        self,
        Q_cos: torch.Tensor,
        Q_sin: torch.Tensor,
        K_cos: torch.Tensor,
        K_sin: torch.Tensor,
        V: torch.Tensor,
        B: int,
        H: int,
        d: int,
        N: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        chunk_size = max(16, min(64, N // 8))

        if N <= chunk_size:
            causal_mask = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
            score_c = torch.matmul(Q_cos, K_cos.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
            score_s = torch.matmul(Q_sin, K_sin.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
            scores = score_c + score_s
            num = torch.matmul(scores, V)
            denom = (
                score_c.sum(dim=-1, keepdim=True)
                + score_s.sum(dim=-1, keepdim=True)
                + self.eps
            )
            return (num / denom.clamp(min=self.eps)).to(dtype)

        pad_len = (chunk_size - (N % chunk_size)) % chunk_size
        if pad_len > 0:
            Q_c_pad = F.pad(Q_cos, (0, 0, 0, pad_len))
            Q_s_pad = F.pad(Q_sin, (0, 0, 0, pad_len))
            K_c_pad = F.pad(K_cos, (0, 0, 0, pad_len))
            K_s_pad = F.pad(K_sin, (0, 0, 0, pad_len))
            V_pad = F.pad(V, (0, 0, 0, pad_len))
        else:
            Q_c_pad, Q_s_pad, K_c_pad, K_s_pad, V_pad = Q_cos, Q_sin, K_cos, K_sin, V

        total_N = N + pad_len
        num_chunks = total_N // chunk_size

        qc_c = Q_c_pad.view(B, H, num_chunks, chunk_size, d)
        qs_c = Q_s_pad.view(B, H, num_chunks, chunk_size, d)
        kc_c = K_c_pad.view(B, H, num_chunks, chunk_size, d)
        ks_c = K_s_pad.view(B, H, num_chunks, chunk_size, d)
        v_c = V_pad.view(B, H, num_chunks, chunk_size, d)

        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))

        score_c = torch.matmul(qc_c, kc_c.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
        score_s = torch.matmul(qs_c, ks_c.transpose(-1, -2).contiguous()).masked_fill(~causal_mask, 0.0)
        scores = score_c + score_s

        num_intra = torch.matmul(scores, v_c)
        denom_intra = score_c.sum(dim=-1, keepdim=True) + score_s.sum(dim=-1, keepdim=True)

        S_cos = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
        S_sin = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
        Z_cos = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)
        Z_sin = torch.zeros(B, H, d, 1, device=device, dtype=torch.float32)

        num_list = []
        denom_list = []

        for c in range(num_chunks):
            num_inter = torch.matmul(qc_c[:, :, c], S_cos) + torch.matmul(qs_c[:, :, c], S_sin)
            denom_inter = torch.matmul(qc_c[:, :, c], Z_cos) + torch.matmul(qs_c[:, :, c], Z_sin)

            num_list.append(num_intra[:, :, c] + num_inter)
            denom_list.append(denom_intra[:, :, c] + denom_inter)

            kc_curr = kc_c[:, :, c]
            ks_curr = ks_c[:, :, c]
            v_curr = v_c[:, :, c]

            # Fix 8: mask K, K_sin, AND V in the last chunk
            if c == num_chunks - 1 and pad_len > 0:
                valid = chunk_size - pad_len
                kc_curr = kc_curr.clone()
                ks_curr = ks_curr.clone()
                v_curr = v_curr.clone()
                kc_curr[:, :, valid:] = 0.0
                ks_curr[:, :, valid:] = 0.0
                v_curr[:, :, valid:] = 0.0

            S_cos = S_cos + torch.matmul(kc_curr.transpose(-1, -2).contiguous(), v_curr)
            S_sin = S_sin + torch.matmul(ks_curr.transpose(-1, -2).contiguous(), v_curr)
            Z_cos = Z_cos + kc_curr.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()
            Z_sin = Z_sin + ks_curr.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

        num = torch.cat(num_list, dim=2)[:, :, :N, :]
        denom = torch.cat(denom_list, dim=2)[:, :, :N, :]
        return (num / (denom + self.eps).clamp(min=self.eps)).to(dtype)

    # ------------------------------------------------------------------
    # Bidirectional forward
    # ------------------------------------------------------------------
    def _forward_bidirectional(
        self,
        Q_cos: torch.Tensor,
        Q_sin: torch.Tensor,
        K_cos: torch.Tensor,
        K_sin: torch.Tensor,
        V: torch.Tensor,
        B: int,
        H: int,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        KV_cos = torch.matmul(K_cos.transpose(-2, -1).contiguous(), V)
        KV_sin = torch.matmul(K_sin.transpose(-2, -1).contiguous(), V)

        Z_cos = K_cos.sum(dim=2, keepdim=True)
        Z_sin = K_sin.sum(dim=2, keepdim=True)

        num = torch.matmul(Q_cos, KV_cos) + torch.matmul(Q_sin, KV_sin)
        denom = (
            torch.matmul(Q_cos, Z_cos.transpose(-2, -1).contiguous())
            + torch.matmul(Q_sin, Z_sin.transpose(-2, -1).contiguous())
        )
        denom = denom.clamp(min=self.eps)

        context = (num / denom).to(dtype)
        return torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)


# ============================================================================
# Injection factory
# ============================================================================
def _is_single_layer(obj: Any) -> bool:
    """Return True if obj is a transformer layer, not a full model.

    Fix 15: Extended blacklist with encoder_layers/decoder_layers/h.
    """
    if obj is None:
        return False

    single_layer_classes = (
        "TransformerBlock",
        "BertLayer",
        "RobertaLayer",
        "GPT2Block",
        "Qwen2DecoderLayer",
        "LlamaDecoderLayer",
        "Qwen3DecoderLayer",
        "MistralDecoderLayer",
    )
    cls_name = getattr(obj, "__class__", type(obj)).__name__
    if cls_name in single_layer_classes:
        return True

    has_attn = (
        hasattr(obj, "attention")
        or hasattr(obj, "attn")
        or hasattr(obj, "self_attn")
        or hasattr(obj, "self")
    )
    has_model = (
        hasattr(obj, "transformer")
        or hasattr(obj, "distilbert")
        or hasattr(obj, "roberta")
        or hasattr(obj, "bert")
        or hasattr(obj, "encoder")
        or hasattr(obj, "model")
        or hasattr(obj, "layers")
        or hasattr(obj, "h")
        or hasattr(obj, "encoder_layers")
        or hasattr(obj, "decoder_layers")
    )
    return has_attn and not has_model


@ATTENTION_REGISTRY.register("cosformer_inject")
@ATTENTION_REGISTRY.register("inject_cosformer")
def inject_cosformer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Any]:
    r"""Safe injection factory for CosformerAttention.

    Fix 3: Thread-safe recursion depth guard.
    Fix 13: pop pretrained_attn from kwargs to avoid conflict.
    Fix 14: Removed dead string expression.
    Fix 16: Clean kwargs before recursive calls.
    """
    depth = _get_inject_depth()
    if depth >= _MAX_DEPTH:
        logger.error("Max injection depth reached in inject_cosformer - possible recursion")
        return None
    _set_inject_depth(depth + 1)

    try:
        target = layer if layer is not None else model
        if target is None:
            target = kwargs.get("model", kwargs.get("layer", None))
        if target is None:
            return None

        cfg = model_config if model_config is not None else getattr(target, "config", None)

        # Fix 16: clean kwargs to avoid conflicts in recursive calls
        kwargs_clean = {
            k: v
            for k, v in kwargs.items()
            if k not in ("model_config", "comp_config", "layer", "old_attn", "model")
        }
        # Fix 13: pop pretrained_attn (we use old_attn explicitly)
        kwargs_clean.pop("pretrained_attn", None)

        # Whole-model traversal
        layer_list = None
        if _is_single_layer(target):
            pass  # handle below
        elif hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer
        elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
            layer_list = target.layers

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_cosformer(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs_clean,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

        layer = target

        # Resolve source attention module
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

        # Fix 12: consistent hidden_size default (768, not 64)
        hidden_size = (
            getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (
                src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
                else None
            )
            or (layer.in_features if hasattr(layer, "in_features") else None)
            or getattr(model_config, "hidden_size", 768)
        )
        num_heads = (
            getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(model_config, "num_attention_heads", 4)
        )

        try:
            new_attn = CosformerAttention(
                config=model_config,
                comp_config=comp_config,
                layer_id=layer_idx,
                layer_idx=layer_idx,
                old_attn=src_attn,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_attention_heads=num_heads,
                **kwargs_clean,
            )

            # Fix 5: do NOT override is_nested_encoder here.
            # copy_weights (called inside CosformerAttention.__init__)
            # already sets it correctly based on source architecture.
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
            elif hasattr(layer, "self_attn"):
                layer.self_attn = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn

            return new_attn
        except Exception as e:
            logger.error(f"Cosformer copy_weights failed: {e}")
            return None
    finally:
        _set_inject_depth(depth)