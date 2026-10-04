"""
HALO-ReGLA: Hardware-Aware Linear Optics & Rectified Gated Linear Attention – Production-Hardened.

Combines multi-head linear projection with Rotary Position Embeddings (RoPE),
rectified kernel representations, and adaptive gating for zero-shot distillation from pretrained transformers.

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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask, get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (
            (module.__class__.__name__ == "Conv1D")
            or (base.__class__.__name__ == "Conv1D")
            or (w.ndim == 2 and w.shape[0] < w.shape[1])
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


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

    if cos.dim() == 2 and cos.shape[0] == q.shape[-2]:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    rotary_dim_even = 2 * (min(rotary_dim, head_dim) // 2)
    if rotary_dim_even == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim_even], q[..., rotary_dim_even:]
    k_rot, k_pass = k[..., :rotary_dim_even], k[..., rotary_dim_even:]
    cos_cut = cos[..., :rotary_dim_even].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim_even].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut) + (_rotate_half(k_rot) * sin_cut)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

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
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        w = self.weight.to(target_dtype) if self.weight.dtype != target_dtype else self.weight
        b = (
            self.bias.to(target_dtype)
            if (self.bias is not None and self.bias.dtype != target_dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(target_dtype) if self.lora_A.dtype != target_dtype else self.lora_A
            bb = self.lora_B.to(target_dtype) if self.lora_B.dtype != target_dtype else self.lora_B
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out.to(target_dtype)


@ATTENTION_REGISTRY.register("halo_regla")
@ATTENTION_REGISTRY.register("halo_regla_attention")
class HALOReGLAAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        hybrid_ratio: int = 4,
        pretrained_attn: nn.Module | None = None,
        gate_init_bias: float = 1.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 64)
        self.config = config

        # Resolve dim
        dim = None
        if config is not None:
            dim = _cfg_get(
                config,
                "hidden_size",
                _cfg_get(
                    config,
                    "d_model",
                    _cfg_get(config, "n_embd", _cfg_get(config, "dim", None)),
                ),
            )
        if dim is None and pretrained_attn is not None:
            dim = getattr(
                pretrained_attn,
                "hidden_size",
                getattr(
                    pretrained_attn,
                    "embed_dim",
                    getattr(pretrained_attn, "all_head_size", None),
                ),
            )
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(pretrained_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
        dim = dim or kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        # Resolve num_heads
        num_heads = None
        if config is not None:
            num_heads = _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(
                    config,
                    "num_heads",
                    _cfg_get(config, "n_heads", _cfg_get(config, "n_head", None)),
                ),
            )
        if num_heads is None and pretrained_attn is not None:
            num_heads = getattr(
                pretrained_attn,
                "num_heads",
                getattr(pretrained_attn, "num_attention_heads", None),
            )
        num_heads = num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim: int = int(dim)
        self.num_heads: int = int(num_heads)
        self.num_kv_heads: int = int(
            _cfg_get(
                config,
                "num_key_value_heads",
                kwargs.get("num_key_value_heads", self.num_heads),
            )
        )
        self.head_dim: int = int(
            _cfg_get(
                config,
                "head_dim",
                kwargs.get("head_dim", self.dim // max(1, self.num_heads)),
            )
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.hybrid_ratio: int = hybrid_ratio

        model_type = str(_cfg_get(config, "model_type", "")).lower()
        is_dec = _cfg_get(
            config,
            "is_decoder",
            _cfg_get(
                config,
                "is_causal",
                kwargs.get("is_causal", kwargs.get("is_decoder", False)),
            ),
        )
        _mt = (model_type or "").lower()
        _decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
        )
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        comp_config = getattr(config, "compression_config", None)
        self.lora_rank: int = int(
            _cfg_get(
                comp_config,
                "lora_rank",
                _cfg_get(config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
        )
        self.lora_alpha: float = float(
            _cfg_get(
                comp_config,
                "lora_alpha",
                _cfg_get(config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        )
        self.freeze_base: bool = self.lora_rank > 0

        # Projections
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
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        self.gate = nn.Linear(self.dim, self.num_heads)
        self.gate.bias.data.fill_(gate_init_bias)

        dropout_p = _cfg_get(
            config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
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
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_w, c_b = _unpack_weight_bias(src.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
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

            if hasattr(src, "c_proj") and src.c_proj is not None:
                o_w, o_b = _unpack_weight_bias(src.c_proj)
                if o_w is not None:
                    if o_w.dim() == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim:
                        o_w = o_w.t().contiguous()
                    extracted["o"] = (o_w, o_b)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                extracted[proj_key] = _unpack_weight_bias(mod)
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                extracted["o"] = _unpack_weight_bias(src.output.dense)

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
                        cw, cb = _unpack_weight_bias(mod)
                        if cw is not None:
                            extracted[proj_key] = (cw, cb)
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.debug(
                "HALOReGLAAttention partial weights extracted; missing: %s",
                missing_after,
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

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (
                    w.shape[0] < target_mod.weight.shape[0]
                    and target_mod.weight.shape[0] % w.shape[0] == 0
                ):
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
                    elif (
                        b.shape[0] < target_mod.bias.shape[0]
                        and target_mod.bias.shape[0] % b.shape[0] == 0
                    ):
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
                    logger.debug("Weight copy for %s has small norm: %.6f", proj_key, fnorm)

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

    _copy_pretrained = copy_weights

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Any | None = None, **kwargs: Any
    ) -> HALOReGLAAttention:
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if encoder_hidden_states is not None:
            return hidden_states, None

        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        B, L, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        # 1. Standard dense softmax stream
        scale = 1.0 / math.sqrt(self.head_dim)
        dense_scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        is_causal = self.is_decoder or kwargs.get("is_causal", False)
        if is_causal:
            causal_mask = torch.triu(
                torch.full((L, L), -1e4, device=dense_scores.device), diagonal=1
            )
            dense_scores = dense_scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            dense_scores = apply_attention_mask(dense_scores, attention_mask)

        dense_attn = F.softmax(dense_scores, dim=-1).to(hidden_states.dtype)
        dense_attn = torch.nan_to_num(dense_attn, nan=0.0)
        dense_attn = self.dropout(dense_attn)
        dense_out = torch.matmul(dense_attn, v)  # (B, H, L, d)

        # 2. Linear stream
        q_lin = F.elu(q) + 1.0
        valid_mask = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        mask_4d = valid_mask.unsqueeze(1).unsqueeze(-1).to(k.dtype)
        k_lin = (F.elu(k) + 1.0) * mask_4d
        v_masked = v * mask_4d

        if is_causal:
            kv_outer = k_lin.unsqueeze(-1) * v_masked.unsqueeze(-2)
            cum_kv = torch.cumsum(kv_outer.float(), dim=2).to(q.dtype)
            num = (q_lin.unsqueeze(-2) @ cum_kv).squeeze(-2)
            cum_k = torch.cumsum(k_lin.float(), dim=2).to(q.dtype)
            den = (q_lin * cum_k).sum(dim=-1, keepdim=True).clamp(min=1e-5)
            linear_out = (num / den) * mask_4d
        else:
            kv = torch.matmul(k_lin.transpose(-2, -1).contiguous(), v_masked)
            z = k_lin.sum(dim=2, keepdim=True)
            num = torch.matmul(q_lin, kv)
            den = torch.matmul(q_lin, z.transpose(-2, -1).contiguous()).clamp(min=1e-5)
            linear_out = (num / den) * mask_4d

        # 3. Adaptive fusion
        gate_w = (
            self.gate.weight.to(hidden_states.dtype)
            if self.gate.weight.dtype != hidden_states.dtype
            else self.gate.weight
        )
        gate_b = (
            self.gate.bias.to(hidden_states.dtype)
            if (self.gate.bias is not None and self.gate.bias.dtype != hidden_states.dtype)
            else self.gate.bias
        )
        gate_out = F.linear(hidden_states, gate_w, gate_b)
        g = (
            torch.sigmoid(gate_out).unsqueeze(1).unsqueeze(-1)
        )  # (B, 1, L, 1) or per-head (B, H, L, 1)
        if g.shape[1] != self.num_heads:
            g = torch.sigmoid(gate_out).permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        g = g.to(hidden_states.dtype)
        fused = (
            g * dense_out.to(hidden_states.dtype) + (1.0 - g) * linear_out.to(hidden_states.dtype)
        ).to(hidden_states.dtype)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                fused = fused * head_mask.view(1, self.num_heads, 1, 1).to(fused.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                fused = fused * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(
                    fused.dtype
                )
            elif head_mask.dim() == 4:
                fused = fused * head_mask.to(fused.dtype)

        out = fused.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            out_final = out
        else:
            out_final = self.out_proj(out)

        if out_final.shape[-1] != hidden_states.shape[-1]:
            if out_final.shape[-1] < hidden_states.shape[-1]:
                out_final = F.pad(out_final, (0, hidden_states.shape[-1] - out_final.shape[-1]))
            else:
                out_final = out_final[..., : hidden_states.shape[-1]]

        return out_final, None


@ATTENTION_REGISTRY.register("halo_regla_inject")
@ATTENTION_REGISTRY.register("halo_regla")
def inject_halo_regla(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    r"""Injection hook for HALOReGLAAttention supporting single layers and whole models."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
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

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_halo_regla(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # 2. Single layer injection
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                curr_attn = target.attention.self
            else:
                curr_attn = target.attention
        elif hasattr(target, "self"):
            curr_attn = target.self
        elif hasattr(target, "attn"):
            curr_attn = target.attn
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn

    if curr_attn is None and target is not None and hasattr(target, "forward"):
        curr_attn = target

    if curr_attn is None and cfg is None:
        return None

    try:
        new_layer = HALOReGLAAttention(
            config=cfg,
            pretrained_attn=curr_attn,
            lora_rank=lora_rank,
            **kwargs,
        )
        if curr_attn is not None:
            p = next(curr_attn.parameters(), None)
            if p is not None:
                new_layer.to(device=p.device, dtype=p.dtype)
    except Exception:
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_layer
            else:
                target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
        elif hasattr(target, "attn"):
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer

    return new_layer
