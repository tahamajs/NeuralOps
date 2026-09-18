"""LEKAv1: Linear-time Efficient Kernel Attention (Production-Hardened).

Mathematical Mechanism:
    Approximates the Gaussian RBF kernel via Random Fourier Feature (RFF) mapping:
    Supports:
    - Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
    - Trainable LoRA adapters with frozen base weights
    - GQA with repeat_kv
    - RoPE position embeddings
    - Causal cumulative recurrence for decoders
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

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


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


@ATTENTION_REGISTRY.register("leka_v1", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class LEKAV1Attention(BaseAttention):
    r"""LEKAv1: Linear-time Efficient Kernel Attention (Hardened)."""

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        comp_config: Optional[Any] = None,
        layer_idx: int = 0,
        total_layers: int = 6,
        bert_mode: bool = False,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None:
            if config is not None:
                hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif source_attn is not None:
                out_layer = getattr(source_attn, "out_lin", getattr(source_attn, "dense", getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None))))
                hidden_size = getattr(out_layer, "out_features", 768) if out_layer is not None else 768
            else:
                hidden_size = 768

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            elif source_attn is not None:
                num_heads = getattr(source_attn, "num_attention_heads", getattr(source_attn, "n_heads", 12))
            else:
                num_heads = 12

        if num_kv_heads is None:
            if config is not None:
                num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", num_heads))
            elif source_attn is not None:
                num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                num_kv_heads = num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        self.feature_dim = getattr(comp_config, "feature_dim", 256) if comp_config is not None else 256
        dropout_p = getattr(comp_config, "dropout", getattr(config, "dropout", 0.1)) if config is not None else 0.1

        # Decoder & nested encoder flags
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

        # LoRA config
        self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if config else 16
        self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.query = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.key = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.value = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        self.norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.pres_gate = nn.Parameter(torch.tensor(2.0))
        self.output_scale = nn.Parameter(torch.ones(1))

        # QR-orthogonalised random projection buffer
        proj_data = torch.randn(self.num_heads, self.head_dim, self.feature_dim)
        for h in range(self.num_heads):
            q_h, _ = torch.linalg.qr(proj_data[h].T if self.head_dim < self.feature_dim else proj_data[h])
            if self.head_dim < self.feature_dim:
                proj_data[h] = q_h.T
            else:
                proj_data[h] = q_h[:, :self.feature_dim]
        self.register_buffer("proj", proj_data, persistent=True)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Positive kernel feature embedding \phi(x) = [cos(x W), sin(x W)] / sqrt(F)."""
        x_norm = self.norm(x)
        features = torch.einsum("bhnd,hdf->bhnf", x_norm, self.proj.to(device=x.device, dtype=x.dtype))
        scale = 1.0 / math.sqrt(self.feature_dim)
        phi_cos = torch.cos(features) * scale
        phi_sin = torch.sin(features) * scale
        return torch.cat([phi_cos, phi_sin], dim=-1)

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
                cls_c = getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                base_cls_c = getattr(base_c, "__class__", type(base_c)).__name__
                if "Conv1D" in cls_c or "Conv1D" in base_cls_c or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1] and "Linear" not in cls_c and "Linear" not in base_cls_c):
                    c_w = c_w.t()
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
                    cls_o = getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                    base_cls_o = getattr(base_o, "__class__", type(base_o)).__name__
                    if "Conv1D" in cls_o or "Conv1D" in base_cls_o or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1] and "Linear" not in cls_o and "Linear" not in base_cls_o):
                        o_w = o_w.t()
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
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"LEKAV1Attention copy_weights incomplete: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
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

                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().shape:
                    w = w.t()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"LEKAV1Attention weight verification warning for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

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
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = self.key(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.value(hidden_states).view(B, N, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        Q_features = self.kernel_feature_map(q)  # (B, H, N, 2F)
        K_features = self.kernel_feature_map(k)  # (B, H, N, 2F)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, :, None].to(hidden_states.dtype)
            elif attention_mask.dim() == 4:
                mask = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(1).unsqueeze(-1).to(hidden_states.dtype)
            else:
                mask = attention_mask.to(hidden_states.dtype)
            K_features = K_features * mask
            v = v * mask
        else:
            mask = None

        if self.is_decoder:
            # Causal linear attention via cumulative sum
            kv_outer = torch.einsum("bhnf,bhnd->bhnfd", K_features.float(), v.float())
            KV_cum = torch.cumsum(kv_outer, dim=2)
            num = torch.einsum("bhnf,bhnfd->bhnd", Q_features.float(), KV_cum)
            z_cum = torch.cumsum(K_features.float(), dim=2)
            denom = torch.einsum("bhnf,bhnf->bhn", Q_features.float(), z_cum).unsqueeze(-1).clamp(min=1e-5)
            context = (num / denom).to(hidden_states.dtype)
        else:
            # Bidirectional linear attention
            KV = torch.einsum("bhnf,bhnd->bhfd", K_features.float(), v.float())
            K_sum = K_features.float().sum(dim=2, keepdim=True)
            num = torch.einsum("bhnf,bhfd->bhnd", Q_features.float(), KV)
            denom = (torch.einsum("bhnf,bhf->bhn", Q_features.float(), K_sum.squeeze(2)).unsqueeze(-1)).clamp(min=1e-5)
            context = (num / denom).to(hidden_states.dtype)

        context = torch.nan_to_num(context, nan=0.0)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            context = context * hm

        if mask is not None:
            context = context * mask

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        alpha = torch.sigmoid(self.pres_gate)
        out = alpha * out + (1.0 - alpha) * hidden_states
        return out, None

    def get_ortho_loss(self) -> float:
        return 0.0


@ATTENTION_REGISTRY.register("leka_v1_inject")
def inject_leka_v1(
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
            res = inject_leka_v1(
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

    src_attn = old_attn if old_attn is not None else (getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None))) if layer is not None else None)

    hidden_size = (
        getattr(src_attn, 'hidden_size', None) or
        getattr(src_attn, 'embed_dim', None) or
        (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
        (layer.in_features if hasattr(layer, 'in_features') else None) or
        getattr(model_config, 'hidden_size', 64)
    )
    num_heads = (
        getattr(src_attn, 'num_heads', None) or
        getattr(src_attn, 'num_attention_heads', None) or
        getattr(model_config, 'num_attention_heads', 4)
    )

    try:
        new_attn = LEKAV1Attention(
            config=model_config,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            bert_mode=False,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
        return new_attn
    except Exception:
        return None
