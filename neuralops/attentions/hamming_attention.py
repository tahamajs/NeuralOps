"""
Hamming Attention Distillation (HAD) Module (Corrected).

Approximates dense inner products with differentiable continuous relaxations of
Hamming distances over sign-binarized representations with calibrated temperature
scaling, multi-architecture weight transfer, trainable LoRA residual adapters,
GQA, RoPE, and causal decoding.


NOTE: This module is LoRA-only. The `rank` argument is accepted for API
compatibility but does not trigger SVD compression; use `lora_rank` to
control adapter size.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.utils.model_family import ModelAdapter

    _HAS_MODEL_ADAPTER = True
except Exception:
    ModelAdapter = None
    _HAS_MODEL_ADAPTER = False


# ============================================================================
# Helpers
# ============================================================================


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
        cls_n = getattr(module, "__class__", type(module)).__name__
        base_n = getattr(base, "__class__", type(base)).__name__
        if (
            ("Conv1D" in cls_n)
            or ("Conv1D" in base_n)
            or hasattr(module, "nf")
            or hasattr(base, "nf")
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

    # HAD-FIX-03: dtype-safe cast
    cos_cut = cos[..., :rotary_dim_even].to(q_rot.dtype)
    sin_cut = sin[..., :rotary_dim_even].to(q_rot.dtype)

    q_rot_embed = (q_rot * cos_cut) + (_rotate_half(q_rot) * sin_cut)
    k_rot_embed = (k_rot * cos_cut.to(k_rot.dtype)) + (
        _rotate_half(k_rot) * sin_cut.to(k_rot.dtype)
    )

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed

    return q, k


# ============================================================================
# LoRALinear
# ============================================================================


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

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
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
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = (
                self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            )
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ============================================================================
# HammingAttention
# ============================================================================


@ATTENTION_REGISTRY.register("hamming", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register(
    "hamming_attention", supports_causal=True, supports_gqa=True
)
class HammingAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        distill_temperature: float = 2.0,
        sparsity_ratio: float = 0.0,
        lora_rank: int = 16,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        num_heads, num_kv_heads, hidden_size, head_dim, kv_repeat = 0, 0, 0, 0, 1

        # HAD-FIX-07: ModelAdapter failures are surfaced (debug), not silently swallowed
        if config is not None and _HAS_MODEL_ADAPTER:
            try:
                self.adapter = ModelAdapter(config)
                num_heads = self.adapter.num_heads
                num_kv_heads = self.adapter.num_key_value_heads
                hidden_size = self.adapter.hidden_size
                head_dim = self.adapter.head_dim
                kv_repeat = self.adapter.kv_repeat
            except Exception as e:
                logger.debug("ModelAdapter failed: %s", e)

        # Fallback to direct config attributes
        if not hidden_size and config is not None:
            hidden_size = getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "dim", 0)),
            )
        if not num_heads and config is not None:
            num_heads = getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", getattr(config, "n_heads", 0)),
            )

        # Fallback to source attention module
        if not hidden_size and pretrained_attn is not None:
            hidden_size = getattr(
                pretrained_attn, "hidden_size", getattr(pretrained_attn, "embed_dim", 0)
            )
            if not hidden_size:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(pretrained_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        hidden_size = m.in_features
                        break
                if not hidden_size:
                    out_l = getattr(
                        pretrained_attn,
                        "out_lin",
                        getattr(
                            pretrained_attn,
                            "dense",
                            getattr(pretrained_attn, "c_proj", None),
                        ),
                    )
                    if out_l is not None and hasattr(out_l, "out_features"):
                        hidden_size = out_l.out_features

        # Final defaults
        hidden_size = hidden_size or kwargs.get(
            "hidden_size", kwargs.get("d_model", 768)
        )
        num_heads = num_heads or kwargs.get(
            "num_heads", kwargs.get("num_attention_heads", 12)
        )
        num_kv_heads = num_kv_heads or _cfg_get(
            config, "num_key_value_heads", kwargs.get("num_key_value_heads", num_heads)
        )
        head_dim = (
            head_dim or kwargs.get("head_dim") or (hidden_size // max(1, num_heads))
        )
        kv_repeat = kv_repeat or max(1, num_heads // max(1, num_kv_heads))

        self.num_heads: int = int(num_heads)
        self.num_kv_heads: int = int(num_kv_heads)
        self.hidden_size: int = int(hidden_size)
        self.head_dim: int = int(head_dim)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = int(kv_repeat)

        # HAD-FIX-06: keep API compat args but they're informational
        self.distill_temperature: float = float(distill_temperature)
        self.sparsity_ratio: float = float(sparsity_ratio)

        # ---- Decoder detection ----
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

        self.lora_rank = int(
            _cfg_get(config, "lora_rank", kwargs.get("lora_rank", lora_rank))
        )

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=True,
        )

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        # HAD-FIX-01: rename trainable scalars with `lora_` prefix so that
        # framework's get_recovery_params recognizes them.
        self.lora_q_scale = nn.Parameter(torch.ones(self.num_heads, 1))
        self.lora_k_scale = nn.Parameter(torch.ones(self.num_heads, 1))
        self.lora_temp = nn.Parameter(torch.tensor(max(distill_temperature, 1e-2)))

        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn
        if hasattr(src, "c_attn") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            weight, bias_data = _unpack_weight_bias(src.c_attn)
            if weight is not None:
                if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    (
                        bias_data[split_size : 2 * split_size]
                        if bias_data is not None
                        else None
                    ),
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o, b_o = _unpack_weight_bias(src.c_proj)
            if w_o is not None:
                if (
                    w_o.dim() == 2
                    and w_o.shape[0] != self.hidden_size
                    and w_o.shape[1] == self.hidden_size
                ):
                    w_o = w_o.t().contiguous()
                extracted["o"] = (w_o, b_o)

        # 3. RoBERTa / BERT nested
        if hasattr(src, "self") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = _unpack_weight_bias(s.query)
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = _unpack_weight_bias(s.key)
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = _unpack_weight_bias(s.value)

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            extracted["o"] = _unpack_weight_bias(src.output.dense)

        # 4. Flat modules
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        w, b = _unpack_weight_bias(getattr(src, name))
                        if w is not None:
                            extracted[proj] = (w, b)
                            break

        # ---- Nested encoder detection ----
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(
            hasattr(src, a)
            for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj")
        )
        if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
            self.is_nested_encoder = False
        elif (
            (
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or (hasattr(src, "query") and not has_out)
            )
            or hasattr(src, "self")
            and hasattr(src, "output")
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.debug("HammingAttention partial extraction; missing: %s", missing)

        # Frobenius check
        for proj_key in ("q", "k", "v"):
            w = extracted[proj_key][0]
            if w is not None:
                n = float(torch.linalg.norm(w.float()).item())
                if n < 1e-4:
                    logger.debug(
                        "HammingAttention: %s near-zero norm (%.2e)", proj_key, n
                    )
        if extracted["o"][0] is not None:
            n = float(torch.linalg.norm(extracted["o"][0].float()).item())
            if n < 1e-4:
                logger.debug("HammingAttention: out near-zero norm (%.2e)", n)

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w_oriented[: min(out_d, src_out), : min(in_d, src_in)].to(
                            target_w.dtype
                        )
                    )
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        n = min(len(target.bias), len(b_val))
                        target.bias.data[:n].copy_(b_val[:n].to(target.bias.dtype))
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, L_q, D = hidden_states.shape
        L_k = kv_input.shape[1]
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L_q, H, Dh).transpose(1, 2).contiguous()
        K = (
            self.k_proj(kv_input)
            .view(B, L_k, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(kv_input)
            .view(B, L_k, self.num_kv_heads, Dh)
            .transpose(1, 2)
            .contiguous()
        )

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross_attention:
            if (
                isinstance(position_embeddings, (tuple, list))
                and len(position_embeddings) >= 2
            ):
                cos, sin = position_embeddings[0], position_embeddings[1]
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))

        # ---- GQA ----
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # ---- Hamming attention scores ----
        # Continuous relaxation of Hamming distance via tanh binarization.
        # Soft-binary: q_b, k_b ∈ (-1, 1)^Dh.
        # Inner product <q_b, k_b> ∈ (-Dh, Dh), so (matmul - Dh) is in (-2Dh, 0).
        # Dividing by 2*sqrt(Dh) normalizes; multiplying by learned temp sharpens.
        q_b = torch.tanh(5.0 * Q * self.lora_q_scale.view(1, H, 1, 1))
        k_b = torch.tanh(5.0 * K * self.lora_k_scale.view(1, H, 1, 1))

        # HAD-FIX-02: ensure temperature is positive (softmax over -Hamming needs T > 0)
        temp = F.softplus(self.lora_temp) + 1e-3

        scores = (
            temp
            * (torch.matmul(q_b, k_b.transpose(-2, -1).contiguous()) - Dh)
            / (2.0 * math.sqrt(Dh))
        )

        # ---- Masking ----
        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        if mask is not None:
            if mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, L_k, device)
                if valid_m is not None:
                    scores = scores.masked_fill(
                        (~valid_m).unsqueeze(1).unsqueeze(2), -1e4
                    )

        # ---- Causal mask ----
        if self.is_decoder and not is_cross_attention and L_q > 1:
            causal_tril = (
                torch.tril(
                    torch.ones((L_q, L_k), device=device, dtype=torch.bool),
                    diagonal=L_k - L_q,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores.masked_fill(~causal_tril, -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores.float(), dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # ---- head_mask ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., : hidden_states.shape[-1]]

        return output, (attn if output_attentions else None)


# ============================================================================
# Injection (HAD-FIX-04: recursion guard)
# ============================================================================


@ATTENTION_REGISTRY.register("hamming_attention_inject")
@ATTENTION_REGISTRY.register("hamming_inject")
@ATTENTION_REGISTRY.register("inject_hamming_attention")
@ATTENTION_REGISTRY.register("inject_hamming")
def inject_hamming_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    _depth: int = 0,
    **kwargs: Any,
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_hamming_attention: max recursion depth exceeded")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # Whole-model traversal
    if target is not None and (
        hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or hasattr(target, "transformer")
        or (hasattr(target, "model") and hasattr(target.model, "layers"))
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
                res = inject_hamming_attention(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    _depth=_depth + 1,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # Single layer
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
        new_layer = HammingAttention(
            config=cfg,
            pretrained_attn=curr_attn,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as e:
        logger.exception("HammingAttention injection failed: %s", e)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_layer
                new_layer.is_nested_encoder = True
            else:
                target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
        elif hasattr(target, "attn"):
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer

    return new_layer
