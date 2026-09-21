"""SOA Fused Attention: Low-Rank Latent Space Multi-Head Attention with Zero-Shot SVD Initialization – Production-Hardened.

Factorizes dense Q, K, and V linear projection operators into separate low-rank bottleneck pairs
initialized from truncated Singular Value Decomposition (SVD) of source attention weights.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve configuration value from dict, object, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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

    if rotary_dim > head_dim:
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
        rotary_dim = head_dim

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


class FactorizedLoRAUp(nn.Module):
    """Up-projection with frozen base weight and trainable LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = lora_alpha / lora_rank
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


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
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


class FactorizedLinear(nn.Module):
    def __init__(self, down: nn.Linear, up: nn.Linear) -> None:
        super().__init__()
        self.down = down
        self.up = up

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.up, "bias", None)

    @property
    def in_features(self) -> int:
        return self.down.in_features

    @property
    def out_features(self) -> int:
        return self.up.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


@ATTENTION_REGISTRY.register("soa_fused_module")
@ATTENTION_REGISTRY.register("soa_fused_attn")
@ATTENTION_REGISTRY.register("SOAFusedAttention")
@ATTENTION_REGISTRY.register("soa_fused_attention")
class SOAFusedAttention(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        k_rank: int = 64,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config or kwargs.get("config", None)
        super().__init__(cfg, rank=k_rank, **kwargs)
        self.config = cfg
        comp_config = getattr(cfg, "compression_config", None)

        self.hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "dim", kwargs.get("hidden_size", kwargs.get("dim", 768)))) if cfg else kwargs.get("hidden_size", kwargs.get("dim", 768))
        self.dim = self.hidden_size
        self.num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", getattr(cfg, "num_heads", kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))))) if cfg else kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))
        self.num_kv_heads = getattr(
            cfg,
            "num_key_value_heads",
            getattr(cfg, "num_kv_heads", kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", self.num_heads))),
        ) if cfg else kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", self.num_heads))
        self.num_heads = int(self.num_heads or 12)
        self.num_kv_heads = int(self.num_kv_heads or self.num_heads)

        # Robust head_dim probe: check source weight first
        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", kwargs.get("old_attn", None))
        head_dim = kwargs.get("head_dim", None)
        if not head_dim and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    if mod.weight.ndim == 2:
                        head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                        break
            if not head_dim:
                head_dim = getattr(source_attn, "head_dim", None)
        if not head_dim and cfg is not None:
            head_dim = getattr(cfg, "head_dim", None)
        if not head_dim or head_dim <= 0:
            head_dim = self.dim // max(1, self.num_heads)

        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.num_queries_per_kv = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.kv_repeat = self.num_queries_per_kv

        # Adaptive k_rank: at least 128 (min 128, dim // 4)
        if k_rank is None or k_rank == 64:
            effective_rank = getattr(comp_config, "k_rank", getattr(comp_config, "soa_rank", rank if rank is not None else max(128, self.dim // 4)))
        else:
            effective_rank = min(k_rank, self.dim)
        self.k_rank = effective_rank
        self.rank = effective_rank

        # Robust is_decoder detection with prefix matching
        _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = (
            getattr(cfg, "is_decoder", False)
            or getattr(cfg, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (source_attn is not None and hasattr(source_attn, "c_attn"))
            or (source_attn is not None and getattr(source_attn, "is_causal", False))
            or (source_attn is not None and getattr(source_attn, "is_decoder", False))
            or bool(kwargs.get("is_decoder", False))
            or bool(kwargs.get("is_causal", False))
        )
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = bool(kwargs.get("is_nested_encoder", False))

        self.lora_rank = getattr(cfg, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if cfg else 16
        self.lora_alpha = getattr(cfg, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if cfg else 32.0
        self.freeze_base = (self.lora_rank > 0)

        # Projections: down is frozen linear, up is FactorizedLoRAUp with trainable LoRA
        self.q_down = nn.Linear(self.dim, self.k_rank, bias=False)
        self.k_down = nn.Linear(self.dim, self.k_rank, bias=False)
        self.v_down = nn.Linear(self.dim, self.k_rank, bias=False)

        self.q_up = FactorizedLoRAUp(self.k_rank, self.all_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        self.k_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        self.v_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, bias=False, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)

        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        dropout_p = getattr(cfg, "dropout", getattr(cfg, "attention_probs_dropout_prob", dropout)) if cfg else dropout
        self.dropout = nn.Dropout(dropout_p)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def q_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.q_down, self.q_up)

    @property
    def k_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.k_down, self.k_up)

    @property
    def v_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.v_down, self.v_up)

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
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
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception:
                pass

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"SOAFusedAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )
            return

        with torch.no_grad():
            pairs = [
                ("q", self.q_down, self.q_up, self.all_head_size),
                ("k", self.k_down, self.k_up, self.all_kv_head_size),
                ("v", self.v_down, self.v_up, self.all_kv_head_size),
            ]
            for proj_key, down, up, target_out in pairs:
                w, _ = extracted[proj_key]
                if w is None:
                    continue
                w_f = w.float()
                if w_f.shape[0] != target_out:
                    if w_f.shape[0] < target_out and target_out % w_f.shape[0] == 0:
                        repeats = target_out // w_f.shape[0]
                        w_f = w_f.repeat(repeats, 1)
                    else:
                        full_w = torch.zeros((target_out, w_f.shape[1]), dtype=w_f.dtype, device=w_f.device)
                        m0 = min(target_out, w_f.shape[0])
                        full_w[:m0, :].copy_(w_f[:m0, :])
                        w_f = full_w

                U, S, Vt = torch.linalg.svd(w_f, full_matrices=False)
                S = S.clamp(min=1e-6)
                r = min(self.k_rank, U.size(1), Vt.size(0))

                down.weight.data.zero_()
                up.weight.data.zero_()
                down.weight.data[:r, :].copy_(Vt[:r, :].to(down.weight.dtype))
                up.weight.data[:, :r].copy_((U[:, :r] * S[:r]).to(up.weight.dtype))

                fnorm = torch.linalg.norm(torch.matmul(up.weight, down.weight).float()).item()
                if fnorm < 1e-4:
                    logger.warning(f"Weight copy notice for {proj_key}: norm {fnorm:.6f} < 1e-4")

            w_o, b_o = extracted["o"]
            if w_o is None:
                if self.is_nested_encoder:
                    if hasattr(self.out_proj, "weight") and self.out_proj.weight.ndim == 2:
                        nn.init.eye_(self.out_proj.weight)
                    if hasattr(self.out_proj, "bias") and self.out_proj.bias is not None:
                        nn.init.zeros_(self.out_proj.bias)
            else:
                if w_o.shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o)
                elif w_o.t().contiguous().shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o.t().contiguous())
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o)

                fnorm = torch.linalg.norm(self.out_proj.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(f"Weight copy notice for out_proj: norm {fnorm:.6f} < 1e-4")

        if self.freeze_base:
            for mod in [self.q_down, self.k_down, self.v_down]:
                mod.weight.requires_grad = False
                if getattr(mod, "bias", None) is not None:
                    mod.bias.requires_grad = False
            for mod in [self.q_up, self.k_up, self.v_up]:
                if hasattr(mod, "freeze_base_weights"):
                    mod.freeze_base_weights()
                else:
                    mod.weight.requires_grad = False
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.requires_grad = False
            if hasattr(self.out_proj, "freeze_base_weights"):
                self.out_proj.freeze_base_weights()
            else:
                self.out_proj.weight.requires_grad = False
                if getattr(self.out_proj, "bias", None) is not None:
                    self.out_proj.bias.requires_grad = False

    _init_from_pretrained = copy_weights

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SOAFusedAttention":
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = hidden_states.shape

        q_proj = self.q_up(self.q_down(hidden_states))
        k_proj = self.k_up(self.k_down(hidden_states))
        v_proj = self.v_up(self.v_down(hidden_states))

        q = q_proj.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_proj.view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_proj.view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # GQA repetition if num_kv_heads < num_heads
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.num_queries_per_kv)
            v = repeat_kv(v, self.num_queries_per_kv)

        # RoPE application
        cos = kwargs.get("cos", None)
        sin = kwargs.get("sin", None)
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        scores = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale

        # Causal mask for decoder
        if self.is_decoder:
            causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Attention mask
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask)

        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        context = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            return context, (probs if output_attentions else None)

        out = self.out_proj(context)
        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("soa_fused_attn_inject")
@ATTENTION_REGISTRY.register("soa_fused_inject")
def inject_soa_fused_attn(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    target_layer_idx: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Union[SOAFusedAttention, nn.Module]]:
    """Safely replace attention module with SOAFusedAttention across diverse transformer architectures."""
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
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_soa_fused_attn(
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
    try:
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

        if src_attn is None:
            return None

        # Double-injection guard
        if src_attn is not None:
            if isinstance(src_attn, SOAFusedAttention) or "SOAFused" in src_attn.__class__.__name__:
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, SOAFusedAttention) or "SOAFused" in cand.__class__.__name__:
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        k_rank = getattr(comp_config, "soa_rank", getattr(comp_config, "rank", kwargs.pop("k_rank", 128)))
        dropout = getattr(comp_config, "dropout", kwargs.pop("dropout", 0.1))

        new_attn = SOAFusedAttention(
            config=cfg,
            k_rank=k_rank,
            dropout=dropout,
            pretrained_attn=src_attn,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    target.attention = new_attn
                return new_attn
            for attr in ("self_attn", "attn", "self"):
                if hasattr(target, attr):
                    setattr(target, attr, new_attn)
                    return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_soa_fused_attn failed at layer %d: %s", layer_idx, e)
        return None


inject_soa_fused_module = inject_soa_fused_attn
inject_soafusedattention = inject_soa_fused_attn
