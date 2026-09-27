"""Information-Bottlenecked Jamba Architecture with Dual Operational Modes.

Differential attention with concept-bottlenecked SSM branch.

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from framework.attentions.simple_ssm import SimpleSSM
except Exception as e:
    logger_import = logging.getLogger(__name__)
    logger_import.warning("SimpleSSM import failed: %s", e)
    SimpleSSM = None  # type: ignore

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE (dtype-safe, past_len-aware)
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
        raise ValueError(
            f"apply_rotary_pos_emb: cos/sin seq len {cos.shape[-2]} != q seq len {q.shape[-2]}"
        )

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
# Decoder detection helper
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    src_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
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
        target_dtype = x.dtype
        w = self.weight.to(target_dtype) if self.weight.dtype != target_dtype else self.weight
        b = self.bias.to(target_dtype) if (self.bias is not None and self.bias.dtype != target_dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(target_dtype) if self.lora_A.dtype != target_dtype else self.lora_A
            bb = self.lora_B.to(target_dtype) if self.lora_B.dtype != target_dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out.to(target_dtype)


# ===========================================================================
# Concept Bottleneck
# ===========================================================================

class InfoMambaConceptBottleneck(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.W_down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.W_up = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_up(F.gelu(self.W_down(x)))


# ===========================================================================
# Main JambaInfoBlock
# ===========================================================================

@ATTENTION_REGISTRY.register("jamba_info", supports_causal=True, supports_gqa=True)
class JambaInfoBlock(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        mode: str = "retrieval",
        sparsity: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # JI-03: robust super().__init__
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

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
        self.mode = mode
        self.sparsity = float(sparsity)

        # JI-12: respect explicit is_decoder
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
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
        self.W_q1 = LoRALinear(self.dim, self.all_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_k1 = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_q2 = LoRALinear(self.dim, self.all_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_k2 = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                               lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                               freeze_base=self.freeze_base)
        self.W_v = LoRALinear(self.dim, self.all_kv_head_size, bias=False,
                              lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                              freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        # Aliases
        self.q_proj = self.W_q1
        self.k_proj = self.W_k1
        self.v_proj = self.W_v

        # Concept bottleneck + SSM
        bottleneck_dim = int(getattr(comp_config, "bottleneck_dim", 256) if comp_config else 256)
        self.bottleneck = InfoMambaConceptBottleneck(self.dim, bottleneck_dim=bottleneck_dim)
        if SimpleSSM is not None:
            try:
                self.ssm = SimpleSSM(d_model=self.dim, config=config, is_decoder=self.is_decoder)
            except TypeError:
                try:
                    self.ssm = SimpleSSM(d_model=self.dim, config=config)
                except TypeError:
                    self.ssm = SimpleSSM(self.dim)
        else:
            self.ssm = None
            logger.warning("JambaInfoBlock: SimpleSSM unavailable; SSM branch disabled.")

        # JI-14: alpha_raw = -2.0 ⇒ sigmoid(-2) ≈ 0.12 (12% SSM initial)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.0))

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Dimension detection helpers (JI-11)
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
            for attr in ("q_proj", "q_lin", "query", "W_q1", "c_attn"):
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

    @staticmethod
    def _detect_head_dim(config: Any, src: Optional[nn.Module], kwargs: Dict[str, Any],
                         hidden_size: int, num_heads: int) -> int:
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self") and hasattr(getattr(src, "self"), "query")
                and hasattr(src, "output") and hasattr(getattr(src, "output"), "dense")):
            is_nested = True
        elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention",
                           "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())) \
                and not has_out_lin:
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
                    or (c_w.ndim == 2 and c_w.shape[1] == 3 * self.hidden_size)
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
            logger.warning(
                "JambaInfoBlock.copy_weights: missing %s from %s; keeping defaults.",
                missing_after, src.__class__.__name__,
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        def _copy_into(target_mod: LoRALinear, w, b) -> None:
            """Copy w into target_mod.weight (with orientation and shape padding)."""
            if w is None:
                return
            if w.ndim != 2:
                w = w.reshape(w.size(0), -1)
            if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                w = w.t().contiguous()
            r = min(target_mod.weight.size(0), w.size(0))
            c = min(target_mod.weight.size(1), w.size(1))
            target_mod.weight.data[:r, :c].copy_(
                w[:r, :c].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype)
            )
            if b is not None and target_mod.bias is not None:
                n = min(b.numel(), target_mod.bias.numel())
                target_mod.bias.data[:n].copy_(
                    b[:n].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                )

        with torch.no_grad():
            w_q, _ = extracted["q"]
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_o, b_o = extracted["o"]

            # Q1 (primary query branch)
            if w_q is not None:
                _copy_into(self.W_q1, _orient(w_q, self.hidden_size), None)

            # JI-02: Q2 and K2 initialized from pretrained (differential attention convention)
            if w_q is not None:
                _copy_into(self.W_q2, _orient(w_q, self.hidden_size), None)
            if w_k is not None:
                _copy_into(self.W_k2, _orient(w_k, self.hidden_size), None)

            # K1 (primary key branch)
            if w_k is not None:
                _copy_into(self.W_k1, _orient(w_k, self.hidden_size), None)

            # V
            if w_v is not None:
                _copy_into(self.W_v, _orient(w_v, self.hidden_size), None)

            # O
            if w_o is not None:
                _copy_into(self.out_proj, _orient(w_o, self.all_head_size), b_o)
            elif self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

            # Norm verification
            for name, mod in (("W_q1", self.W_q1), ("W_k1", self.W_k1),
                              ("W_q2", self.W_q2), ("W_k2", self.W_k2),
                              ("W_v", self.W_v), ("out_proj", self.out_proj)):
                w_norm = float(torch.norm(mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        "JambaInfoBlock: weight '%s' has near-zero norm (%.2e); reinit.",
                        name, w_norm,
                    )
                    nn.init.kaiming_uniform_(mod.weight, a=math.sqrt(5))

            if self.freeze_base:
                for mod in (self.W_q1, self.W_k1, self.W_q2, self.W_k2,
                            self.W_v, self.out_proj):
                    mod.freeze_base_weights()

        # JI-09: forward sanity check
        try:
            with torch.no_grad():
                d = next(self.parameters()).device
                dt = self.W_q1.weight.dtype
                dummy = torch.randn(1, 4, self.dim, device=d, dtype=dt)
                _ = self.forward(dummy)
        except Exception as e:
            logger.debug("JambaInfoBlock forward sanity check warning: %s", e)

    # ------------------------------------------------------------------
    # SSM wrapper (JI-08)
    # ------------------------------------------------------------------
    def _run_ssm(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.ssm is None:
            return torch.zeros_like(x)
        try:
            out = self.ssm(x, attention_mask=attention_mask)
        except TypeError:
            try:
                out = self.ssm(x)
            except Exception as e:
                logger.debug("SSM forward fallback failed: %s", e)
                return torch.zeros_like(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

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
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Allow mask passed via kwargs
        mask = kwargs.get("mask", None)
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        device = x.device

        q1 = self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k1 = self.W_k1(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        q2 = self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k2 = self.W_k2(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.W_v(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # JI-04: RoPE with past_len offset
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
                cos = cos[..., past_len: past_len + L, :]
                sin = sin[..., past_len: past_len + L, :]
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)

        if self.kv_repeat > 1:
            k1 = repeat_kv(k1, self.kv_repeat)
            k2 = repeat_kv(k2, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1).contiguous()) * scale
        scores2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
            scores1 = scores1.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)
            scores2 = scores2.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        # JI-07: robust mask handling
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = get_token_valid_mask(attention_mask, B, L, device)
                scores1 = scores1.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
                scores2 = scores2.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    mask_add = (~attention_mask).float() * -1e4
                else:
                    mask_add = attention_mask.float()
                scores1 = scores1 + mask_add
                scores2 = scores2 + mask_add

        scores1 = torch.nan_to_num(scores1, nan=-1e4, posinf=1e4, neginf=-1e4)
        scores2 = torch.nan_to_num(scores2, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn1 = F.softmax(scores1, dim=-1).to(q1.dtype)
        attn2 = F.softmax(scores2, dim=-1).to(q2.dtype)

        # Differential attention
        diff_attn = F.relu(attn1 - 0.2 * attn2)
        diff_attn = diff_attn / (diff_attn.sum(dim=-1, keepdim=True) + 1e-8)
        diff_attn = self.dropout(diff_attn).to(v.dtype)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            diff_attn = diff_attn * hm.to(diff_attn.dtype)

        attn_out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(
            B, L, self.all_head_size
        )

        # SSM branch
        bottleneck_x = self.bottleneck(x)
        if self.is_decoder:
            ssm_branch = self._run_ssm(bottleneck_x, attention_mask)
        else:
            x_rev = torch.flip(bottleneck_x, dims=[1])
            ssm_fwd = self._run_ssm(bottleneck_x, attention_mask)
            ssm_bwd = self._run_ssm(x_rev, attention_mask)
            ssm_branch = (0.5 * (ssm_fwd + torch.flip(ssm_bwd, dims=[1]))).to(x.dtype)

        target_dtype = x.dtype
        alpha = torch.sigmoid(self.alpha_raw).to(target_dtype)
        combined = ((1.0 - alpha) * attn_out.to(target_dtype) + alpha * ssm_branch.to(target_dtype)).to(target_dtype)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = combined
        else:
            out = self.out_proj(combined).to(target_dtype)

        return out.to(target_dtype), (diff_attn if output_attentions else None)


# ===========================================================================
# Injection (JI-05: recursion guard)
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
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_jamba_info: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
            res = inject_jamba_info(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
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

        # Double-injection guard
        if isinstance(src_attn, JambaInfoBlock) or "JambaInfo" in src_attn.__class__.__name__:
            return None

        new_attn = JambaInfoBlock(
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)

        p = next(src_attn.parameters(), None) if src_attn is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if layer is not None:
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            elif hasattr(layer, "attention"):
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            elif hasattr(layer, "self_attn"):
                new_attn.is_nested_encoder = False
                layer.self_attn = new_attn
            elif hasattr(layer, "self"):
                new_attn.is_nested_encoder = True
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                new_attn.is_nested_encoder = False
                layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_jamba_info failed at layer %d: %s", layer_idx, e, exc_info=True)
        return None