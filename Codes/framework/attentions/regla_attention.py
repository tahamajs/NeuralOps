"""Refined Gated Linear Attention (ReGLA) – Production-Hardened.

Combines element-wise key gating, normalized linear attention kernel evaluation,
and short-range depthwise 1D convolutions without O(L) memory explosion.
Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Chunked scan maintaining [B, H, d, d] state
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
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


@ATTENTION_REGISTRY.register("regla")
@ATTENTION_REGISTRY.register("regla_attention")
class ReGLAAttention(BaseAttention):
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        kernel_size: int = 3,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None) if config else None

        self.dim = _cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = _cfg_get(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = _cfg_get(config, "num_key_value_heads", kwargs.get("num_key_value_heads", self.num_heads))

        if self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} not divisible by num_heads={self.num_heads}")

        self.head_dim = self.dim // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kernel_size = kernel_size

        model_type = str(_cfg_get(config, "model_type", "")).lower()
        is_dec = _cfg_get(config, "is_decoder", kwargs.get("is_decoder", False))
        if model_type in ["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama", "mistral"]:
            is_dec = True
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = _cfg_get(config, "lora_rank", _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha = _cfg_get(config, "lora_alpha", _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base = (self.lora_rank > 0)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Projections
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.gate_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # Depthwise 1D conv branch
        self.conv = nn.Conv1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=kernel_size,
            groups=self.dim,
            bias=True,
        )

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        if source_attn is not None:
            self.copy_weights(source_attn)

    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    def get_ortho_loss(self) -> float:
        return 0.0

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
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
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
                        extracted[k] = (w.data.clone(), b.data.clone() if b is not None else None)
            except Exception:
                pass

        with torch.no_grad():
            for proj_key, target_mod in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
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
                    raise RuntimeError(f"ReGLAAttention weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

            # Initialize gate_proj from k_proj
            self.gate_proj.weight.data.copy_(self.k_proj.weight.data)
            if self.k_proj.bias is not None and self.gate_proj.bias is not None:
                self.gate_proj.bias.data.copy_(self.k_proj.bias.data)
            if self.freeze_base:
                self.gate_proj.freeze_base_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("ReGLAAttention does not support cross-attention")

        x = hidden_states
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        g = self.gate_proj(x).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)
            g = repeat_kv(g, repeats)

        # Non-negative kernel activation
        q = F.elu(q) + 1.0
        k_gated = (F.elu(k) + 1.0) * torch.sigmoid(g)

        valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        if valid_mask is None:
            valid_mask = torch.ones(B, L, device=hidden_states.device, dtype=torch.bool)

        mask_4d = valid_mask.unsqueeze(1).unsqueeze(-1).to(k_gated.dtype)
        k_gated = k_gated * mask_4d
        v = v * mask_4d

        is_causal = self.is_decoder or kwargs.get("is_causal", False)

        if is_causal:
            # Chunked linear scan maintaining state [B, H, d, d]
            chunk = 64
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=q.dtype)
            Z = torch.zeros(B, self.num_heads, self.head_dim, device=x.device, dtype=q.dtype)
            out_list = []

            for start in range(0, L, chunk):
                end = min(start + chunk, L)
                q_c = q[:, :, start:end]
                k_c = k_gated[:, :, start:end]
                v_c = v[:, :, start:end]

                intra_kv = torch.cumsum(k_c.unsqueeze(-1) * v_c.unsqueeze(-2), dim=2)
                num_intra = (q_c.unsqueeze(-2) @ intra_kv).squeeze(-2)
                num_inter = (q_c.unsqueeze(-2) @ S.unsqueeze(2)).squeeze(-2)
                num = num_intra + num_inter

                z_intra = torch.cumsum(k_c, dim=2)
                den = (q_c * (z_intra + Z.unsqueeze(2))).sum(-1, keepdim=True).clamp(min=1.0)
                out_list.append(num / den)

                S = S + (k_c.unsqueeze(-1) * v_c.unsqueeze(-2)).sum(dim=2)
                Z = Z + k_c.sum(dim=2)

            out = torch.cat(out_list, dim=2)
        else:
            kv = torch.matmul(k_gated.transpose(-2, -1), v)
            z = k_gated.sum(dim=2, keepdim=True)
            num = torch.matmul(q, kv)
            den = torch.matmul(q, z.transpose(-2, -1)).clamp(min=1.0)
            out = num / den

        out = out * mask_4d

        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                out = out * head_mask.view(1, self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                out = out * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(out.dtype)
            elif head_mask.dim() == 3:
                out = out * head_mask.unsqueeze(-1).to(out.dtype)
            elif head_mask.dim() == 4:
                out = out * head_mask.to(out.dtype)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        # Depthwise 1D conv branch
        pad_2d = valid_mask.unsqueeze(-1).to(x.dtype)
        x_in = (x * pad_2d).permute(0, 2, 1)
        if is_causal:
            x_pad = F.pad(x_in, (self.kernel_size - 1, 0))
            loc = self.conv(x_pad).permute(0, 2, 1)
        else:
            p = self.kernel_size // 2
            x_pad = F.pad(x_in, (p, p))
            loc = self.conv(x_pad).permute(0, 2, 1)

        if loc.shape[-1] != out.shape[-1]:
            loc_proj = loc[..., :out.shape[-1]] if loc.shape[-1] > out.shape[-1] else F.pad(loc, (0, out.shape[-1] - loc.shape[-1]))
        else:
            loc_proj = loc

        fused = (out + loc_proj) * pad_2d

        if self.is_nested_encoder:
            output = fused if fused.shape[-1] == self.dim else self.out_proj(fused)
            return output, None

        output = self.out_proj(fused)
        if output.shape[-1] != hidden_states.shape[-1]:
            output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1])) if output.shape[-1] < hidden_states.shape[-1] else output[..., :hidden_states.shape[-1]]
        return output, None


@ATTENTION_REGISTRY.register("regla_inject")
@ATTENTION_REGISTRY.register("regla_attention_inject")
@ATTENTION_REGISTRY.register("inject_regla")
def inject_regla(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    """Safely replace attention in model or layer with ReGLAAttention."""
    try:
        model = layer
        cfg = model_config or getattr(model, "config", None)
        src = old_attn
        if src is None and model is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(model, attr) and not isinstance(getattr(model, attr), (nn.Dropout, nn.LayerNorm)):
                    candidate = getattr(model, attr)
                    if hasattr(candidate, "query") or hasattr(candidate, "q_lin") or hasattr(candidate, "q_proj") or hasattr(candidate, "c_attn"):
                        src = candidate
                        break

        hidden_size = _cfg_get(cfg, "hidden_size", getattr(src, "hidden_size", 768))
        num_heads = _cfg_get(cfg, "num_attention_heads", getattr(src, "num_attention_heads", 12))

        new_attn = ReGLAAttention(
            config=cfg,
            pretrained_attn=src,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs,
        )

        if model is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(model, attr) and getattr(model, attr) is src:
                    setattr(model, attr, new_attn)
                    return new_attn
            if hasattr(model, "attention"):
                if hasattr(model.attention, "self"):
                    model.attention.self = new_attn
                else:
                    model.attention = new_attn
                return new_attn
            return new_attn

        return new_attn
    except Exception as e:
        logger.error("inject_regla failed: %s", e)
        return None


inject_regla_attention = inject_regla
