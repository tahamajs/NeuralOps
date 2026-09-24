"""Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block (Production-Hardened).

Combines:
- SVD-Mamba selective state space branch
- Multi-Head Attention branch with GQA, RoPE, and causal masking
- Expert Choice MoE branch with auxiliary loss tracking

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

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.expert_choice_moe import ExpertChoiceMoE
from framework.attentions.simple_ssm import SimpleSSM
from framework.attentions.svd_mamba import SVDMamba
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


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


class SVDMambaSub(nn.Module):
    is_custom_attention: bool = True
    def __init__(self, d_model: int = 768, config: Optional[Any] = None, rank: Optional[int] = None, is_decoder: bool = False, **kwargs: Any) -> None:
        super().__init__()

        if d_model is None and config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
        d_model = int(d_model or 768)
        self.d_model = d_model
        self.ssm = SimpleSSM(d_model=d_model, config=config, is_decoder=is_decoder)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> Tuple[torch.Tensor, None]:
        x_proj = self.in_proj(hidden_states)
        ssm_out = self.ssm(x_proj, attention_mask=attention_mask)
        if isinstance(ssm_out, tuple):
            ssm_out = ssm_out[0]

        out = self.out_proj(ssm_out)
        return out, None


@ATTENTION_REGISTRY.register("jamba", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("jamba2", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class Jamba2Block(BaseAttention):
    r"""Jamba-2 Hybrid Architecture combining SVD-Mamba, Multi-Head Attention, and Expert Choice MoE."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_experts: int = 4,
        top_k: int = 2,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim") or d_model
        n_heads = kwargs.get("n_heads") or kwargs.get("num_heads") or kwargs.get("num_attention_heads") or n_heads

        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
            elif source_attn is not None:
                out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                d_model = getattr(out_layer, "out_features", getattr(source_attn, "hidden_size", getattr(source_attn, "dim", 768))) if out_layer is not None else 768
            else:
                d_model = 768

        if n_heads is None:
            if config is not None:
                n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            elif source_attn is not None:
                n_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
            else:
                n_heads = 12

        if config is not None:
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)

        self.d_model = int(d_model)
        self.dim = self.d_model
        self.hidden_size = self.d_model
        self.n_heads = int(n_heads)
        self.num_heads = self.n_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads)) if config else self.num_heads
        self.head_dim = self.d_model // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Decoder detection
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
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False


        # LoRA config: independent inspection of config, comp_config, and kwargs
        if config is not None:
            self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank = kwargs.get("lora_rank", 16)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base = (self.lora_rank > 0)
        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        # MHA Projections
        self.q_lin = LoRALinear(self.d_model, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.d_model, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.d_model, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin

        # Branch 1: SSM
        self.mamba = SVDMambaSub(self.d_model, config=config, rank=rank, is_decoder=self.is_decoder)

        # Branch 3: MoE
        self.moe = ExpertChoiceMoE(self.d_model, n_experts=n_experts, capacity_factor=1.0)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss = torch.tensor(0.0)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def freeze_base_weights(self) -> None:
        """Freeze base linear projection weights when LoRA adapters are present."""
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "freeze_base_weights"):
                mod.freeze_base_weights()

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base linear projection weights."""
        for mod in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
            if hasattr(mod, "unfreeze_base_weights"):
                mod.unfreeze_base_weights()

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, '__class__', type(src)).__name__
        has_out = hasattr(src, 'dense') or hasattr(src, 'out_proj') or hasattr(src, 'out_lin') or hasattr(src, 'c_proj') or hasattr(src, 'o_proj')
        if (cls_name in ['BertSelfAttention', 'RobertaSelfAttention'] or (hasattr(src, 'query') and not has_out)):
            self.is_nested_encoder = True
        if hasattr(src, 'self') and hasattr(src, 'output'):
            self.is_nested_encoder = True

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
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or hasattr(c_attn_mod, "nf") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
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
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or hasattr(c_proj_mod, "nf") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
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
                            w_data = mod.weight.data.clone()
                            if (mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf") or (w_data.ndim == 2 and proj_key == "o" and w_data.shape[0] < w_data.shape[1]):
                                w_data = w_data.t().contiguous()
                            extracted[proj_key] = (w_data, mod.bias.data.clone() if mod.bias is not None else None)
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
                logger.debug(f"UniversalAdapter fallback: {e}")

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"Jamba2Block copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
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

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(w[:out_d, :in_d]).item())
                if w_norm < 1e-4:
                    raise RuntimeError(f"Jamba2Block weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if self.lora_rank > 0 or self.freeze_base:
            self.freeze_base_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape

        # Branch 1: SVD-Mamba
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask)
        x1 = self.norm1(hidden_states + mamba_out)

        # Branch 2: Multi-Head Attention
        q = self.q_lin(x1).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(x1).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        elif self.rotary_emb is not None or (self.source_attn is not None and hasattr(self.source_attn, "rotary_emb")):
            rotary_mod = self.rotary_emb or getattr(self.source_attn, "rotary_emb", None)
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(k, seq_len=L)
            except TypeError:
                try:
                    cos, sin = rotary_mod(k, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(pos_ids)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Memory-efficient: use SDPA to avoid OOM on large seqs

        try:
            _qs = q.half() if q.dtype == torch.float32 else q
            _ks = k.half() if k.dtype == torch.float32 else k
            _vs = v.half() if v.dtype == torch.float32 else v
            _is_causal = (getattr(self, 'is_decoder', False) or getattr(self, 'is_causal', False) or getattr(self.config, 'is_decoder', False) or getattr(self.config, 'is_causal', False)) and L > 1
            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                _qs, _ks, _vs,
                is_causal=_is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attn_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_proj = attn_out if self.is_nested_encoder else self.out_lin(attn_out)
            x2 = self.norm2(x1 + self.dropout(attn_proj))
            moe_out = self.moe(x2)
            x3 = self.norm3(x2 + self.dropout(moe_out))
            self.aux_loss = getattr(self.moe, "aux_loss", torch.tensor(0.0))
            return x3, None
        except Exception:
            pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder or getattr(self, "is_causal", False) or getattr(self.config, "is_decoder", False) or getattr(self.config, "is_causal", False):
            causal_mask = torch.tril(torch.ones(L, L, device=hidden_states.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e4)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
                scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.clamp(min=-1e4)

        probs = F.softmax(scores, dim=-1).to(v.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            probs = probs * hm

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if not self.is_nested_encoder:
            attn_proj = self.out_lin(attn_out)
        else:
            attn_proj = attn_out

        x2 = self.norm2(x1 + self.dropout(attn_proj))

        # Branch 3: Expert Choice MoE
        moe_out = self.moe(x2)
        x3 = self.norm3(x2 + self.dropout(moe_out))
        self.aux_loss = getattr(self.moe, "aux_loss", torch.tensor(0.0))

        return x3, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("jamba_inject")
@ATTENTION_REGISTRY.register("jamba2_inject")
@ATTENTION_REGISTRY.register("inject_jamba")
@ATTENTION_REGISTRY.register("inject_jamba2")
def inject_jamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[Jamba2Block, nn.Module]]:
    r"""Safe injection factory for Jamba2Block across diverse transformer architectures."""
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
            res = inject_jamba(
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

    try:
        new_block = Jamba2Block(
            d_model=hidden_size,
            config=cfg,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Jamba2Block injection failed: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_block
                new_block.is_nested_encoder = True
            else:
                target.attention = new_block
        elif hasattr(target, "self"):
            target.self = new_block
        elif hasattr(target, "attn"):
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            target.self_attn = new_block

    return new_block
