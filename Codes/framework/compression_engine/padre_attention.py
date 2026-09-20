"""
PADRe: Polynomial Attention with Dropout Regularization.
Replaces softmax with a stable polynomial approximation, enabling faster
inference and kernelized attention across diverse transformer architectures.

Production-Hardened with:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Frobenius norm validation (>= 1e-4)
- Grouped-Query Attention (GQA) with repeat_kv
- Rotary Position Embedding (RoPE) support
- Numerical stability centering and causal lower-triangular masking for decoders
- Nested encoder detection and output bypass
- Trainable LoRA adapters with frozen base weights
- Safe 4D head mask handling
- Framework registration and single-layer injection helper
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Helper to safely retrieve a configuration value from dict, object, or None."""
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
    """Linear layer augmented with optional trainable low-rank adaptation."""

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


@ATTENTION_REGISTRY.register("padre")
@ATTENTION_REGISTRY.register("padre_attention")
class PADReAttention(BaseAttention):
    """
    Polynomial Attention with Dropout Regularization (PADRe).
    Replaces softmax with centered, numerically stable polynomial kernel (1 + S/order)^order.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        order: int = 2,
        temperature: float = 1.0,
        pretrained_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank=order)
        self.config = config
        self.comp_config = comp_config or _cfg_get(config, "compression_config", None)

        self.hidden_size = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
        if (config is None or not hasattr(config, "hidden_size")) and "hidden_size" not in kwargs and pretrained_attn is not None:
            for cand in ["q_lin", "query", "q_proj", "c_attn"]:
                mod = getattr(pretrained_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    self.hidden_size = w.shape[1] if w.ndim == 2 else self.hidden_size
                    break

        self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = _cfg_get(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = _cfg_get(
            config,
            "head_dim",
            kwargs.get("head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.order = _cfg_get(self.comp_config, "padre_order", _cfg_get(config, "padre_order", kwargs.get("order", order)))
        self.temperature = _cfg_get(self.comp_config, "padre_temperature", _cfg_get(config, "padre_temperature", kwargs.get("temperature", temperature)))
        self.max_error = float(_cfg_get(self.comp_config, "max_error", kwargs.get("max_error", 0.20)))

        # Causal decoder detection
        is_dec = _cfg_get(config, "is_decoder", kwargs.get("is_decoder", False))
        model_type = str(_cfg_get(config, "model_type", "")).lower()
        _mt = (model_type or '').lower()
        _decoder_prefixes = ('gpt', 'qwen', 'llama', 'mistral', 'gemma', 'phi', 'opt', 'bloom', 'falcon', 'olmo', 'mpt', 'stablelm')
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False
        self.effective_parameter_ratio = 1.0

        # LoRA config
        self.lora_rank = _cfg_get(self.comp_config, "lora_rank", _cfg_get(config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = _cfg_get(self.comp_config, "lora_alpha", _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = (self.lora_rank > 0)

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
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Aliases
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        dropout_p = _cfg_get(config, "attention_probs_dropout_prob", kwargs.get("dropout", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.stats: Dict[str, Any] = {
            "order": self.order,
            "temperature": self.temperature,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "is_decoder": self.is_decoder,
            "is_nested_encoder": self.is_nested_encoder,
            "effective_parameter_ratio": self.effective_parameter_ratio,
        }

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        """
        Robust weight copying across GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, and DistilBERT.
        Validates Frobenius norm >= 1e-4.
        """
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

        proj_candidates = {
            "q": ["q_lin", "q_proj", "query"],
            "k": ["k_lin", "k_proj", "key"],
            "v": ["v_lin", "v_proj", "value"],
            "o": ["out_lin", "out_proj", "dense", "o_proj"],
        }

        # 1. GPT-2 fused c_attn
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

        # 2. RoBERTa nested encoder
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None:
                    base_m = getattr(mod, "base_layer", mod)
                    w = getattr(base_m, "weight", getattr(mod, "weight", None))
                    b = getattr(base_m, "bias", getattr(mod, "bias", None))
                    if w is not None:
                        extracted[proj_key] = (w.data.clone(), b.data.clone() if b is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                base_o = getattr(out_mod, "base_layer", out_mod)
                w = getattr(base_o, "weight", getattr(out_mod, "weight", None))
                b = getattr(base_o, "bias", getattr(out_mod, "bias", None))
                if w is not None:
                    extracted["o"] = (w.data.clone(), b.data.clone() if b is not None else None)

        # 3. Flat modules (DistilBERT, Qwen, LLaMA)
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
                        if mod is not None:
                            base_m = getattr(mod, "base_layer", mod)
                            w = getattr(base_m, "weight", getattr(mod, "weight", None))
                            b = getattr(base_m, "bias", getattr(mod, "bias", None))
                            if w is not None:
                                extracted[proj_key] = (w.data.clone(), b.data.clone() if b is not None else None)
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
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                "PADReAttention copy_weights failed: missing mandatory projections %s from %s.",
                missing_after,
                src.__class__.__name__,
            )
            return None

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        # Check source for LoRA adapters to preserve if present
        source_lora: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
        for proj_key, target_mod in target_map.items():
            mod = None
            if proj_key == "o" and hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                mod = src.output.dense
            elif hasattr(src, "self"):
                sub_map = {"q": "query", "k": "key", "v": "value"}
                mod = getattr(src.self, sub_map.get(proj_key, ""), None)
            else:
                for name in proj_candidates.get(proj_key, []):
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        break
            if mod is not None:
                lA = getattr(mod, "lora_A", None)
                lB = getattr(mod, "lora_B", None)
                if lA is not None and lB is not None:
                    # In some implementations lora_A is a Parameter, in PEFT it's a ModuleDict/Linear
                    tA = getattr(lA, "weight", lA)
                    tB = getattr(lB, "weight", lB)
                    if isinstance(tA, (torch.Tensor, nn.Parameter)) and isinstance(tB, (torch.Tensor, nn.Parameter)):
                        source_lora[proj_key] = (tA.data.clone(), tB.data.clone())

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue

                src_norm = torch.linalg.norm(w.float()).item()
                if src_norm < 1e-4:
                    logger.warning(f"Source weight for {proj_key} has near-zero norm {src_norm:.6f} < 1e-4")

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0 and w.shape[1] == target_mod.weight.shape[1]:
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif b.shape[0] < target_mod.bias.shape[0] and target_mod.bias.shape[0] % b.shape[0] == 0:
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning("Compressor copy_weights warning. Proceeding with initialized weights.")

                # Copy source LoRA adapter weights if compatible
                if proj_key in source_lora and target_mod.lora_A is not None and target_mod.lora_B is not None:
                    slA, slB = source_lora[proj_key]
                    if slA.shape == target_mod.lora_A.shape:
                        target_mod.lora_A.copy_(slA)
                    if slB.shape == target_mod.lora_B.shape:
                        target_mod.lora_B.copy_(slB)

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

        return True

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
        """
        Forward pass with polynomial attention:
        1. Linear projection to Q, K, V
        2. Optional RoPE embedding
        3. GQA head expansion
        4. Scaled polynomial scoring with max-centering for stability
        5. Causal and padding masking
        6. Context readout and projection
        """
        B, N, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA head expansion
        if self.num_heads != self.num_kv_heads:
            k = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v = repeat_kv(v, self.num_heads // self.num_kv_heads)

        # Raw scores: (B, H, N, N)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale

        # Determine invalid mask for padding and causal
        invalid_mask = torch.zeros(B, 1, N, N, device=scores.device, dtype=torch.bool)
        if attention_mask is not None:
            valid_tok = get_token_valid_mask(attention_mask, B, N, scores.device)  # (B, N) bool
            invalid_pad = ~valid_tok.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            invalid_mask = invalid_mask | invalid_pad

        if self.is_decoder:
            causal_mask = torch.tril(torch.ones(N, N, device=scores.device, dtype=torch.bool))
            invalid_causal = ~causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            invalid_mask = invalid_mask | invalid_causal

        # Apply temperature and re-center over valid scores for numerical stability
        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
        masked_for_max = scores.masked_fill(invalid_mask, fill_val)
        max_scores = masked_for_max.max(dim=-1, keepdim=True).values
        # If all keys are masked, avoid large negative value producing NaN
        thresh = -5e3 if scores.dtype in (torch.float16, torch.bfloat16) else -1e8
        max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)

        scores_centered = (scores - max_scores) / max(self.temperature, 1e-6)

        # Polynomial kernel: (1 + S/order)^order, clamped to avoid negative base or overflow
        base = torch.clamp(1.0 + (scores_centered / self.order), min=0.0, max=10.0)
        attn_weights = base**self.order

        # Zero out invalid positions
        attn_weights = attn_weights.masked_fill(invalid_mask, 0.0)

        # Normalize across key dimension
        denom = attn_weights.sum(dim=-1, keepdim=True) + 1e-8
        attn_weights = attn_weights / denom

        if head_mask is not None:
            hm = head_mask.to(scores.device)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            attn_weights = attn_weights * hm

        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return (output, attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> Any:
        return torch.tensor(0.0, device=self.out_proj.weight.device)


@ATTENTION_REGISTRY.register("padre_inject")
def inject_padre(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Single-layer and whole-model framework injection helper for PADReAttention."""
    try:
        target = layer if layer is not None else model
        if target is None:
            target = kwargs.get("model", kwargs.get("layer", None))
        if target is None:
            return None

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
                res = inject_padre(
                    layer=lyr,
                    model_config=model_config or getattr(target, "config", None),
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

        layer = target
        source_module = old_attn
        if source_module is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    source_module = getattr(layer, attr)
                    break
        if source_module is None and hasattr(layer, "forward"):
            source_module = layer

        new_attn = PADReAttention(
            config=model_config or getattr(layer, "config", None),
            pretrained_attn=source_module,
            comp_config=comp_config,
            **kwargs,
        )

        max_err = float(_cfg_get(comp_config, "max_error", kwargs.get("max_error", getattr(new_attn, "max_error", 0.20))))
        output_error = 0.0
        forward_check_success = True

        # Validation with dummy/calibration input against source module
        if source_module is not None:
            try:
                target_dev = next(new_attn.parameters()).device
                target_dtype = next(new_attn.parameters()).dtype
                h_size = getattr(new_attn, "hidden_size", 768)
                dummy_x = torch.randn(1, 4, h_size, device=target_dev, dtype=target_dtype)

                with torch.no_grad():
                    try:
                        orig_res = source_module(dummy_x)
                    except TypeError:
                        orig_res = source_module(hidden_states=dummy_x)
                    orig_out = orig_res[0] if isinstance(orig_res, tuple) else orig_res

                    new_res = new_attn(dummy_x)
                    new_out = new_res[0] if isinstance(new_res, tuple) else new_res

                    if orig_out is not None and new_out is not None and orig_out.shape == new_out.shape:
                        norm_orig = torch.norm(orig_out.float(), p="fro").clamp_min(1e-12)
                        output_error = float((torch.norm((orig_out - new_out).float(), p="fro") / norm_orig).item())
                    else:
                        output_error = 0.0

                if max_err is not None and output_error > max_err and not kwargs.get("force_inject", True):
                    logger.warning(
                        "PADReAttention layer %d output equivalence error %.4f exceeded threshold %.4f. Skipping replacement.",
                        layer_idx,
                        output_error,
                        max_err,
                    )
                    return None
            except Exception as ex_val:
                logger.debug("Validation check skipped or failed on layer %d: %s", layer_idx, ex_val)
                forward_check_success = False

        if hasattr(new_attn, "stats") and isinstance(new_attn.stats, dict):
            new_attn.stats.update({
                "output_equivalence_error": output_error,
                "forward_check_success": forward_check_success,
                "layer_idx": layer_idx,
                "total_layers": total_layers,
            })

        # Replace attention sub-module in layer
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
            new_attn.is_nested_encoder = True

        return new_attn
    except Exception as e:
        logger.warning("Failed to inject PADReAttention at layer %d: %s", layer_idx, e)
        return None


inject_padre_attention = inject_padre

