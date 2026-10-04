"""
Information-Bottlenecked Jamba Architecture with Dual Operational Modes.
Differential attention with concept-bottlenecked SSM branch.

=== FIXES APPLIED (2026-10-04) ===
FIX 1: (1 - λ_init) scaling per Diff Transformer paper (Ye et al., ICLR 2025).
FIX 2: Per-head λ_init via the paper's formula: 0.8 - 0.6 * exp(-0.3 * h).
FIX 3: RMSNorm applied to SSM branch for scale consistency with attention.
FIX 4: Correct mask reversal in encoder bidirectional SSM.
FIX 5: position_ids fallback via rotary_emb when position_embeddings is None.
FIX 6: Robust cache dtype handling (no per-step .to() calls).
FIX 7: dtype-consistent sigmoid on alpha_raw.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralops.attentions.simple_ssm import SimpleSSM
except Exception as e:
    logging.getLogger(__name__).warning("SimpleSSM import failed: %s", e)
    SimpleSSM = None  # type: ignore

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE & Utilities
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
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    seq_axis = -2
    if cos.shape[seq_axis] != q.shape[seq_axis]:
        if cos.shape[seq_axis] >= q.shape[seq_axis]:
            offset = cos.shape[seq_axis] - q.shape[seq_axis]
            cos = cos.narrow(seq_axis, offset, q.shape[seq_axis])
            sin = sin.narrow(seq_axis, offset, q.shape[seq_axis])
        else:
            raise ValueError(
                f"apply_rotary_pos_emb: cos seq len {cos.shape[seq_axis]} < "
                f"q seq len {q.shape[seq_axis]}"
            )

    rotary_dim, head_dim = cos.shape[-1], q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q, sin_q = cos.to(q_rot.dtype), sin.to(q_rot.dtype)
        cos_k, sin_k = cos.to(k_rot.dtype), sin.to(k_rot.dtype)
        q_rot = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
        k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    else:
        cos_q, sin_q = cos[..., :head_dim].to(q.dtype), sin[..., :head_dim].to(q.dtype)
        cos_k, sin_k = cos[..., :head_dim].to(k.dtype), sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        b, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(b, num_kv_heads * n_rep, slen, head_dim)


def _detect_decoder(
    config: Any, is_decoder: bool | None, src_attn: nn.Module | None
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if mt.startswith((
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )):
            return True
    if src_attn is not None:
        if (
            getattr(src_attn, "is_decoder", False)
            or hasattr(src_attn, "c_attn")
            or "gpt" in src_attn.__class__.__name__.lower()
        ):
            return True
    return False


def _detect_return_format(source_attn: nn.Module | None, is_nested: bool) -> str:
    if is_nested:
        return "bert"
    if source_attn is None:
        return "llama"
    cls = source_attn.__class__.__name__.lower()
    if any(k in cls for k in ("gpt2", "gptne", "gptj", "gpt_neo", "gpt_j")) or hasattr(
        source_attn, "c_attn"
    ):
        return "gpt2"
    return "llama"


def _read_cache(
    past_key_value: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
    if past_key_value is None:
        return None, None, None, 0
    if isinstance(past_key_value, (tuple, list)):
        if len(past_key_value) == 0:
            return None, None, None, 0
        first = past_key_value[0]
        if isinstance(first, (tuple, list)):
            entry = past_key_value[0]
            return entry[0], entry[1], entry[2], entry[0].shape[-2]
        if len(past_key_value) >= 3:
            return (
                past_key_value[0],
                past_key_value[1],
                past_key_value[2],
                past_key_value[0].shape[-2],
            )
    if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
        k1 = past_key_value.key_cache[0]
        k2 = past_key_value.key_cache[1] if len(past_key_value.key_cache) > 1 else k1
        v = (
            past_key_value.value_cache[0]
            if hasattr(past_key_value, "value_cache") and past_key_value.value_cache
            else None
        )
        if k1 is not None:
            return k1, k2, v, k1.shape[-2]
    return None, None, None, 0


# ===========================================================================
# Core Modules
# ===========================================================================
class RMSNorm(nn.Module):
    """RMSNorm for stabilizing Differential Attention Outputs."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


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
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.lora_rank, self.lora_alpha = int(lora_rank), float(lora_alpha)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if (
            self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


class InfoMambaConceptBottleneck(nn.Module):
    """
    Information Bottleneck with down/up projections.
    SSM executes in the compressed (bottleneck_dim) space.
    """

    def __init__(self, dim: int, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.norm_in = nn.LayerNorm(dim)
        self.W_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.norm_mid = nn.LayerNorm(bottleneck_dim)
        self.W_up = nn.Linear(bottleneck_dim, dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)

    def down(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(x)
        z = self.W_down(h)
        return self.norm_mid(z)

    def up(self, z: torch.Tensor) -> torch.Tensor:
        z = F.gelu(z)
        h = self.W_up(z)
        return self.norm_out(h)


@ATTENTION_REGISTRY.register("jamba_info", supports_causal=True, supports_gqa=True)
class JambaInfoBlock(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        is_decoder: bool | None = None,
        per_head_alpha: bool = True,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = self._detect_hidden_size(config, source_attn, kwargs)
        num_heads = self._detect_num_heads(config, source_attn, kwargs, hidden_size)
        num_kv_heads = self._detect_num_kv_heads(config, source_attn, kwargs, num_heads)
        head_dim = self._detect_head_dim(config, source_attn, kwargs, hidden_size, num_heads)

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self._return_format = _detect_return_format(source_attn, self.is_nested_encoder)

        # Optional rotary_emb (for position_ids fallback)
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn else None

        # LoRA
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

        # ---------- Projections ----------
        self.W_q1 = LoRALinear(
            self.dim, self.all_head_size, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k1 = LoRALinear(
            self.dim, self.all_kv_head_size, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_q2 = LoRALinear(
            self.dim, self.all_head_size, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_k2 = LoRALinear(
            self.dim, self.all_kv_head_size, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.W_v = LoRALinear(
            self.dim, self.all_kv_head_size, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Aliases
        self.q_proj, self.k_proj, self.v_proj = self.W_q1, self.W_k1, self.W_v

        # ---------- Differential Variables ----------
        self.lambda_q1 = nn.Parameter(torch.zeros(self.num_heads))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.num_heads))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.num_heads))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.num_heads))

        # FIX 2: Per-head λ_init via Diff Transformer paper formula.
        # Formula: λ_init[h] = 0.8 - 0.6 · exp(-0.3 · h)
        head_ids = torch.arange(self.num_heads, dtype=torch.float32)
        lambda_init_values = 0.8 - 0.6 * torch.exp(-0.3 * head_ids)
        self.lambda_init = nn.Parameter(lambda_init_values)

        # FIX 1: Precompute (1 - λ_init) as a buffer for scale factor.
        # Stored as buffer (not parameter) because it's derived from lambda_init.
        # Recomputed in forward to stay in sync if lambda_init changes.
        # (No need to register as buffer — computed on the fly in forward.)

        # FIX 3: RMSNorm on both branches for scale consistency
        self.diff_norm = RMSNorm(self.all_head_size)
        self.ssm_norm = RMSNorm(self.dim)

        # ---------- Alpha ----------
        self.per_head_alpha = bool(per_head_alpha)
        if self.per_head_alpha:
            self.alpha_raw = nn.Parameter(torch.full((self.num_heads,), -2.0))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(-2.0))

        # ---------- Bottleneck + SSM ----------
        bottleneck_dim = int(
            getattr(comp_config, "bottleneck_dim", 256) if comp_config else 256
        )
        # Ensure bottleneck_dim is reasonable relative to dim
        bottleneck_dim = max(64, min(bottleneck_dim, self.dim))
        self.bottleneck = InfoMambaConceptBottleneck(self.dim, bottleneck_dim=bottleneck_dim)

        # SSM in bottleneck space
        self.ssm_fwd, self.ssm_bwd = None, None
        if SimpleSSM is not None:
            self.ssm_fwd = self._make_ssm(config, bottleneck_dim)
            if not self.is_decoder:
                self.ssm_bwd = self._make_ssm(config, bottleneck_dim)

        dropout_p = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def _make_ssm(self, config: Any | None, d_model: int) -> nn.Module | None:
        if SimpleSSM is None:
            return None
        try:
            return SimpleSSM(d_model=d_model, config=config, is_decoder=self.is_decoder)
        except Exception as e:
            logger.warning("SimpleSSM instantiation failed: %s", e)
            return None

    @staticmethod
    def _detect_hidden_size(config, src, kwargs) -> int:
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
            for attr in ("q_proj", "q_lin", "query", "W_q1", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    return int(mod.in_features)
        return 768

    @staticmethod
    def _detect_num_heads(config, src, kwargs, hidden_size) -> int:
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
        return max(1, hidden_size // 64)

    @staticmethod
    def _detect_num_kv_heads(config, src, kwargs, num_heads) -> int:
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

    @staticmethod
    def _detect_head_dim(config, src, kwargs, hidden_size, num_heads) -> int:
        v = kwargs.get("head_dim")
        if isinstance(v, int) and v > 0:
            return v
        if config is not None:
            for k in ("head_dim", "kv_channels", "d_kv"):
                v = getattr(config, k, None)
                if isinstance(v, int) and v > 0:
                    return v
        if src is not None:
            for attr in ("q_proj", "query", "q_lin", "W_q1"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    try:
                        return int(mod.weight.shape[0] // max(1, num_heads))
                    except Exception:
                        pass
        return max(1, hidden_size // max(1, num_heads))

    def _run_ssm(
        self,
        ssm: nn.Module | None,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if ssm is None:
            return torch.zeros_like(x)
        try:
            out = ssm(x, attention_mask=attention_mask)
        except TypeError:
            try:
                out = ssm(x)
            except Exception as e:
                logger.debug("SSM forward failed: %s", e)
                return torch.zeros_like(x)
        return out[0] if isinstance(out, tuple) else out

    def _format_output(
        self,
        out: torch.Tensor,
        attn_weights: torch.Tensor | None,
        present: Any | None,
    ) -> tuple:
        fmt = self._return_format
        if fmt == "bert":
            return (out, attn_weights)
        if fmt == "gpt2":
            return (out, present, attn_weights)
        return (out, attn_weights, present)

    def _normalize_head_mask(
        self, head_mask: torch.Tensor, attn: torch.Tensor
    ) -> torch.Tensor:
        hm = head_mask.to(attn.dtype)
        H = self.num_heads
        if hm.dim() == 1:
            hm = hm.view(1, H, 1, 1) if hm.numel() == H else hm.view(1, -1, 1, 1)
        elif hm.dim() == 2:
            if hm.shape[-1] == H:
                hm = hm.view(hm.shape[0], H, 1, 1)
            elif hm.shape[0] == H:
                hm = hm.t().view(1, H, 1, 1)
            else:
                hm = hm.view(1, hm.shape[-1], 1, 1)
        elif hm.dim() == 3:
            if hm.shape[1] == H:
                hm = hm.unsqueeze(2)
            elif hm.shape[2] == H:
                hm = hm.permute(0, 2, 1).unsqueeze(2)
            else:
                hm = hm.unsqueeze(1)
        return attn * hm

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return
        cls_name = src.__class__.__name__
        has_out_lin = any(hasattr(src, a) for a in ("out_lin", "c_proj", "o_proj"))
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name in [
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
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c = src.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                if (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (cw.ndim == 2 and cw.shape[1] == 3 * self.hidden_size)
                ):
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )
            if hasattr(src, "c_proj") and src.c_proj is not None:
                o = src.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    if (
                        o.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    ):
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # BERT/RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                d = src.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # Flat
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
                            w = m.weight.data.clone()
                            if m.__class__.__name__ == "Conv1D" or (
                                key == "o" and w.ndim == 2 and w.shape[0] < w.shape[1]
                            ):
                                w = w.t().contiguous()
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                w,
                                b.data.clone() if b is not None else None,
                            )
                            break

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
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback: %s", e)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _copy_into(
            target_mod: LoRALinear, w: torch.Tensor | None, b: torch.Tensor | None
        ) -> None:
            if w is None:
                return
            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)
            if (
                w.shape != target_mod.weight.shape
                and target_mod.weight.shape == w.t().contiguous().shape
            ):
                w = w.t().contiguous()
            r = min(target_mod.weight.size(0), w.size(0))
            c = min(target_mod.weight.size(1), w.size(1))
            target_mod.weight.data[:r, :c].copy_(
                w[:r, :c].to(
                    device=target_mod.weight.device, dtype=target_mod.weight.dtype
                )
            )
            if b is not None and target_mod.bias is not None:
                n = min(b.numel(), target_mod.bias.numel())
                b_val = b.data if isinstance(b, nn.Parameter) else b
                target_mod.bias.data[:n].copy_(
                    b_val[:n].to(
                        device=target_mod.bias.device, dtype=target_mod.bias.dtype
                    )
                )

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            if w_q is not None:
                _copy_into(self.W_q1, _orient(w_q, self.hidden_size), None)
                _copy_into(self.W_q2, _orient(w_q, self.hidden_size), None)
            if w_k is not None:
                _copy_into(self.W_k1, _orient(w_k, self.hidden_size), None)
                _copy_into(self.W_k2, _orient(w_k, self.hidden_size), None)
            if w_v is not None:
                _copy_into(self.W_v, _orient(w_v, self.hidden_size), None)
            if w_o is not None:
                _copy_into(self.out_proj, _orient(w_o, self.all_head_size), b_o)
            elif self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

        if self.freeze_base:
            for mod in (
                self.W_q1,
                self.W_k1,
                self.W_q2,
                self.W_k2,
                self.W_v,
                self.out_proj,
            ):
                mod.freeze_base_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        *args: Any,
        **kwargs: Any,
    ):
        # ---- Input validation ----
        mask = kwargs.get("mask")
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        device = x.device

        # ---- Read cache ----
        past_k1, past_k2, past_v, past_len = _read_cache(past_key_value)

        # ---- QKV projections ----
        q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k1 = (
            self.W_k1(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k2 = (
            self.W_k2(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.W_v(x)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        position_embeddings = position_embeddings or kwargs.get("position_embeddings")
        if position_embeddings is None:
            position_ids = position_ids or kwargs.get("position_ids")

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if cos.shape[-2] >= past_len + L:
                cos = cos.narrow(-2, past_len, L)
                sin = sin.narrow(-2, past_len, L)
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
        elif position_ids is not None and self.rotary_emb is not None:
            # FIX 5: fallback via rotary_emb if available
            try:
                cos, sin = self.rotary_emb(x, position_ids=position_ids)
                if cos.shape[-2] >= L:
                    cos = cos[..., :L, :]
                    sin = sin[..., :L, :]
                q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
                q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            except Exception as e:
                logger.debug("rotary_emb fallback failed: %s", e)

        # ---- KV cache append (before GQA) ----
        if past_k1 is not None:
            # FIX 6: cache is stored in the same dtype as k1
            k1 = torch.cat([past_k1, k1], dim=-2)
            k2 = torch.cat([past_k2, k2], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present_key_value = (k1, k2, v)

        # ---- GQA expand ----
        if self.kv_repeat > 1:
            k1 = repeat_kv(k1, self.kv_repeat)
            k2 = repeat_kv(k2, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Attention scores ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1).contiguous()) * scale
        scores2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1).contiguous()) * scale

        # ---- Causal mask ----
        if self.is_decoder and L > 1:
            kv_len = k1.shape[-2]
            causal = torch.tril(
                torch.ones(L, kv_len, device=device, dtype=torch.bool),
                diagonal=past_len,
            )
            scores1 = scores1.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
            scores2 = scores2.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

        # ---- Attention mask ----
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = get_token_valid_mask(attention_mask, B, k1.shape[-2], device)
                if valid is not None:
                    scores1 = scores1.masked_fill(~valid.unsqueeze(1).unsqueeze(2), -1e4)
                    scores2 = scores2.masked_fill(~valid.unsqueeze(1).unsqueeze(2), -1e4)
            elif attention_mask.dim() == 3:
                m3 = attention_mask.unsqueeze(1).float()
                if attention_mask.dtype == torch.bool:
                    m3 = (~attention_mask).unsqueeze(1).float() * -1e4
                scores1 = scores1 + m3
                scores2 = scores2 + m3
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    m4 = (~attention_mask).float() * -1e4
                else:
                    m4 = attention_mask.float()
                scores1 = scores1 + m4
                scores2 = scores2 + m4

        scores1 = torch.nan_to_num(scores1, nan=-1e4, posinf=1e4, neginf=-1e4)
        scores2 = torch.nan_to_num(scores2, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn1 = F.softmax(scores1, dim=-1).to(q1.dtype)
        attn2 = F.softmax(scores2, dim=-1).to(q2.dtype)

        # ---- Differential Attention (Diff Transformer) ----
        lambda_full = (
            torch.exp(self.lambda_q1 * self.lambda_k1)
            - torch.exp(self.lambda_q2 * self.lambda_k2)
            + self.lambda_init
        ).to(attn1.dtype)
        lambda_full = lambda_full.view(1, self.num_heads, 1, 1)

        # FIX 1: (1 - λ_init) scaling per Diff Transformer paper
        scale_factor = (1.0 - self.lambda_init).to(attn1.dtype).view(
            1, self.num_heads, 1, 1
        )

        diff_attn = (attn1 - lambda_full * attn2) * scale_factor
        diff_attn = self.dropout(diff_attn).to(v.dtype)

        if head_mask is not None:
            diff_attn = self._normalize_head_mask(head_mask, diff_attn)

        attn_out = (
            torch.matmul(diff_attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, self.all_head_size)
        )
        attn_out = self.diff_norm(attn_out)

        # ---- Information Bottleneck + SSM ----
        # NOTE: bottleneck operates on x (dual path with attention).
        # To connect bottleneck to attention, replace `x` with `attn_out` below.
        z = self.bottleneck.down(x)

        if self.is_decoder:
            z_ssm = self._run_ssm(self.ssm_fwd, z, attention_mask)
        else:
            if self.ssm_bwd is not None:
                z_rev = torch.flip(z, dims=[1]).contiguous()
                # FIX 4: proper mask reversal — flip along sequence dim
                if attention_mask is not None:
                    # Determine sequence dim: for (B, L) it's dim=1;
                    # for (B, 1, L, L) it's dim=-1; for (B, L, L) it's dim=-1
                    if attention_mask.dim() == 2:
                        rev_mask = torch.flip(attention_mask, dims=[1]).contiguous()
                    elif attention_mask.dim() == 3:
                        rev_mask = torch.flip(attention_mask, dims=[-1]).contiguous()
                    elif attention_mask.dim() == 4:
                        rev_mask = torch.flip(attention_mask, dims=[-1]).contiguous()
                    else:
                        rev_mask = attention_mask
                else:
                    rev_mask = None

                ssm_fwd = self._run_ssm(self.ssm_fwd, z, attention_mask)
                ssm_bwd = self._run_ssm(self.ssm_bwd, z_rev, rev_mask)
                z_ssm = 0.5 * (ssm_fwd + torch.flip(ssm_bwd, dims=[1]).contiguous())
            else:
                z_ssm = self._run_ssm(self.ssm_fwd, z, attention_mask)

        ssm_branch = self.bottleneck.up(z_ssm)

        # FIX 3: RMSNorm on SSM branch for scale consistency
        ssm_branch = self.ssm_norm(ssm_branch)

        # ---- Alpha blending ----
        target_dtype = x.dtype

        if self.per_head_alpha:
            # FIX 7: sigmoid in target dtype
            alpha = torch.sigmoid(self.alpha_raw.to(target_dtype)).view(
                1, 1, self.num_heads, 1
            )
            attn_out_h = attn_out.view(B, L, self.num_heads, self.head_dim)
            ssm_branch_h = ssm_branch.view(B, L, self.num_heads, self.head_dim)
            combined = (1.0 - alpha) * attn_out_h + alpha * ssm_branch_h
            combined = combined.view(B, L, self.all_head_size)
        else:
            alpha = torch.sigmoid(self.alpha_raw.to(target_dtype))
            combined = (1.0 - alpha) * attn_out + alpha * ssm_branch

        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = combined.to(target_dtype)
        else:
            out = self.out_proj(combined).to(target_dtype)

        attn_weights = diff_attn if output_attentions else None
        return self._format_output(out, attn_weights, present_key_value)


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("jamba_info_inject")
@ATTENTION_REGISTRY.register("inject_jamba_info")
def inject_jamba_info(
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
        return None
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None
    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
        for idx, lyr in enumerate(layer_list):
            res = inject_jamba_info(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=len(layer_list),
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        src_attn = old_attn
        if src_attn is None and layer is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    cand = getattr(layer, attr)
                    src_attn = getattr(cand, "self", cand)
                    break

        if isinstance(src_attn, JambaInfoBlock) or "JambaInfo" in src_attn.__class__.__name__:
            return None

        new_attn = JambaInfoBlock(
            config=cfg, comp_config=comp_config, old_attn=src_attn, **kwargs
        )

        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        if layer is not None:
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
                return None
            new_attn._return_format = _detect_return_format(
                src_attn, new_attn.is_nested_encoder
            )

        return new_attn
    except Exception as e:
        logger.error("inject_jamba_info failed at layer %d: %s", layer_idx, e, exc_info=True)
        return None


__all__ = [
    "JambaInfoBlock",
    "InfoMambaConceptBottleneck",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_jamba_info",
]