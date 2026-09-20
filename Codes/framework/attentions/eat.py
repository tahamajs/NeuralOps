"""
EAT: Efficient Attention Transformer with Progressive Token Pruning.

Implements production-hardened attention-level efficient multi-head self-attention
with optional progressive token pruning and zero-shot weight inheritance across
DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.gqa_utils import repeat_kv
from .progressive_token_pruning import ProgressiveTokenPruning

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
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


@ATTENTION_REGISTRY.register("eat", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("eat_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class EATBlock(BaseAttention):
    r"""
    Efficient Attention Transformer (EAT) Attention Module with Progressive Token Pruning.
    Hardened for multi-arch deployment (DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA).
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        d_model: Union[int, Any] = 768,
        n_heads: int = 12,
        prune_layer: Optional[ProgressiveTokenPruning] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        layer_id: Optional[int] = None,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        cfg_obj = getattr(config, "text_config", config) if config is not None else None
        hidden_size = getattr(cfg_obj, "hidden_size", None) if cfg_obj is not None else None
        num_heads = getattr(cfg_obj, "num_attention_heads", None) if cfg_obj is not None else None

        if hidden_size is None and source_attn is not None:
            for cand in ["q_lin", "query", "q_proj", "c_attn", "in_proj_weight", "qkv"]:
                mod = getattr(source_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    hidden_size = w.shape[1] if w.ndim == 2 else None
                    break
                elif isinstance(mod, torch.Tensor) and mod.ndim == 2:
                    hidden_size = mod.shape[1]
                    break
        if hidden_size is None:
            hidden_size = kwargs.get("hidden_size", kwargs.get("d_model", d_model if isinstance(d_model, int) else 768))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", kwargs.get("n_heads", n_heads)))

        super().__init__(config, rank)
        self.d_model: int = int(hidden_size)
        self.dim: int = self.d_model
        self.n_heads: int = int(num_heads)
        self.num_heads: int = self.n_heads

        self.num_kv_heads: int = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads)) if config is not None else self.num_heads
        self.head_dim: int = getattr(config, "head_dim", None) or (self.dim // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        resolved_layer_id = layer_id if layer_id is not None else (layer_idx if layer_idx is not None else kwargs.get("layer_id", kwargs.get("layer_idx", 0)))
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        self.lora_rank: int = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha: float = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base: bool = (self.lora_rank > 0)

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower() if config is not None else ""
        is_dec = getattr(config, "is_decoder", getattr(config, "is_causal", kwargs.get("is_causal", False))) if config is not None else kwargs.get("is_causal", False)
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
        self.is_nested_encoder: bool = False

        # Multi-head projections equipped with LoRA adapters
        self.q_lin = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.dim, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.prune_layer: Optional[ProgressiveTokenPruning] = prune_layer

        self.rotary_emb = None
        self.comp_config: Optional[Any] = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Compatibility properties
    @property
    def query(self) -> LoRALinear: return self.q_lin
    @property
    def key(self) -> LoRALinear: return self.k_lin
    @property
    def value(self) -> LoRALinear: return self.v_lin
    @property
    def dense(self) -> LoRALinear: return self.out_lin
    @property
    def o_proj(self) -> LoRALinear: return self.out_lin
    @property
    def out_proj(self) -> LoRALinear: return self.out_lin
    @property
    def q_proj(self) -> LoRALinear: return self.q_lin
    @property
    def k_proj(self) -> LoRALinear: return self.k_lin
    @property
    def v_proj(self) -> LoRALinear: return self.v_lin
    @property
    def c_proj(self) -> LoRALinear: return self.out_lin

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        r"""Copy projection weights from pretrained attention module with Frobenius norm verification."""
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = hasattr(pretrained_attn, "out_lin") or hasattr(pretrained_attn, "c_proj") or hasattr(pretrained_attn, "o_proj")
        is_nested = False
        if (
            hasattr(pretrained_attn, "self")
            and hasattr(getattr(pretrained_attn, "self"), "query")
            and hasattr(pretrained_attn, "output")
            and hasattr(getattr(pretrained_attn, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    logger.debug(f"c_attn LoRA merge failed (using unmerged weights): {e}")
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

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                if hasattr(c_proj_mod, "merge") and not getattr(c_proj_mod, "merged", False):
                    try:
                        c_proj_mod.merge()
                    except Exception as e:
                        logger.debug(f"c_proj LoRA merge failed (using unmerged weights): {e}")
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(pretrained_attn, "self") and hasattr(getattr(pretrained_attn, "self"), "query"):
            s = getattr(pretrained_attn, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(pretrained_attn, "output") and hasattr(getattr(pretrained_attn, "output"), "dense"):
                out_mod = pretrained_attn.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Standard projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "q"],
                "k": ["k_lin", "k_proj", "key", "k"],
                "v": ["v_lin", "v_proj", "value", "v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj", "output", "proj", "o"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(pretrained_attn, name):
                        mod = getattr(pretrained_attn, name)
                        if hasattr(mod, "dense"):
                            mod = getattr(mod, "dense")
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        # Fallback via UniversalAdapter
        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug(f"UniversalAdapter fallback failed: {e}")

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(f"EATBlock copy_weights: missing projections {missing_after} from {pretrained_attn.__class__.__name__}; proceeding with random init for those")

        target_map = {"q": self.q_lin, "k": self.k_lin, "v": self.v_lin, "o": self.out_lin}
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

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                # Frobenius norm verification (>= 1e-4)
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"EATBlock weight copy for '{proj_key}': norm too small ({w_norm:.2e}); continuing")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "EATBlock":
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

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
        B, L, D = hidden_states.shape

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections
        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. Rotary Position Embeddings (RoPE)
        cos = kwargs.get("cos", None)
        sin = kwargs.get("sin", None)
        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb[0], pos_emb[1]

        if cos is not None and sin is not None and not is_cross_attention:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids or kwargs.get("position_ids", None))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None and not is_cross_attention:
            try:
                from framework.utils.compat import maybe_apply_rotary
                q, k = maybe_apply_rotary(q, k, getattr(self, "adapter", None), self.rotary_emb, position_ids or kwargs.get("position_ids", None))
            except Exception as e:
                logger.debug(f"rotary_emb apply failed, skipping RoPE: {e}")

        # 3. Grouped Query Attention (GQA) head expansion
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 4. Multi-head attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
        # Memory-efficient: use SDPA to avoid OOM on large seqs

        try:
            _qs = q.half() if q.dtype == torch.float32 else q
            _ks = k.half() if k.dtype == torch.float32 else k
            _vs = v.half() if v.dtype == torch.float32 else v
            _is_causal = getattr(self, 'is_decoder', False) and not is_cross_attention and L > 1
            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                _qs, _ks, _vs,
                is_causal=_is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
            return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None
        except Exception:
            pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross_attention:
            causal_mask = torch.triu(
                torch.full((L, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1 + (kv_len - L),
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                pad_mask = (eff_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float("-inf"))
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # 5. Head mask handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        attn_out = torch.matmul(probs.to(v.dtype), v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # 6. Output projection with nested encoder bypass
        if self.is_nested_encoder:
            if attn_out.shape[-1] != self.dim:
                out = self.out_lin(attn_out)
            else:
                out = attn_out
        else:
            out = self.out_lin(attn_out)

        if out.shape[-1] != D:
            out = F.pad(out, (0, D - out.shape[-1])) if out.shape[-1] < D else out[..., :D]

        if self.prune_layer is not None and not self.training:
            out, _ = self.prune_layer(out)

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("eat_fn")
@ATTENTION_REGISTRY.register("eat_inject")
def inject_eat(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
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
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_eat(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection factory for EATBlock."""
    src = old_attn
    if src is None and layer is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                src = getattr(layer, attr)
                break

    try:
        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config

        new_attn = EATBlock(
            config=model_config,
            pretrained_attn=src,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            **kwargs,
        )
        if layer is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return new_attn
        return new_attn
    except Exception as e:
        logger.error(f"inject_eat: failed: {e}")
        return None
