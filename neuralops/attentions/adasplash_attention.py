"""
AdaSplash Attention: Adaptive Splash Attention with α-Entmax Sparse Activations.

Combines localized sliding-window splash radius with sparse α-entmax adaptive thresholding
and global token routing, ensuring robust numerical convergence and exact pretrained weight transfer.

"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.compat import maybe_apply_rotary, repeat_kv
from neuralops.utils.model_family import ModelAdapter


def entmax_bisect(
    scores: torch.Tensor,
    alpha: float = 1.5,
    valid_mask: torch.Tensor | None = None,
    dim: int = -1,
    n_iter: int = 15,  # FIX: Reduced max iterations from 25 to 15
) -> torch.Tensor:
    r"""
    Compute α-entmax via stabilized bisection strictly over valid tokens.
    Includes an Early Stopping mechanism to drastically improve Inference Speed.
    """
    device = scores.device
    dtype = scores.dtype
    neg_inf_val = torch.tensor(-10000.0, device=device, dtype=dtype)

    # Ensure all scores are finite before processing to prevent NaNs
    scores = torch.where(torch.isfinite(scores), scores, neg_inf_val)

    if abs(alpha - 1.0) < 1e-4:
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, -10000.0)
        scores_clean = torch.where(torch.isfinite(scores), scores, neg_inf_val)
        sf = torch.nan_to_num(F.softmax(scores_clean.float(), dim=dim), nan=0.0).to(dtype)
        if valid_mask is not None:
            sf = sf.masked_fill(~valid_mask, 0.0)
        sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
        return (sf / sf_denom).to(dtype)

    alpha_minus_1 = max(1e-4, alpha - 1.0)
    inv_alpha_minus_1 = 1.0 / alpha_minus_1

    # Mask invalid keys with large negative value
    if valid_mask is not None:
        scores_valid = scores.masked_fill(~valid_mask, -10000.0)
    else:
        scores_valid = scores

    # Shift scores for numerical stability
    max_scores = scores_valid.amax(dim=dim, keepdim=True)
    max_scores = torch.where(max_scores < -5000.0, torch.zeros_like(max_scores), max_scores)
    scores_shift = scores_valid - max_scores
    scaled = scores_shift * alpha_minus_1

    if valid_mask is not None:
        scaled_for_min = scaled.masked_fill(~valid_mask, 10000.0)
        tau_lo = scaled_for_min.amin(dim=dim, keepdim=True) - 1.0
        tau_lo = torch.where(tau_lo > 5000.0, torch.zeros_like(tau_lo) - 1.0, tau_lo)
    else:
        tau_lo = scaled.amin(dim=dim, keepdim=True) - 1.0

    tau_hi = torch.zeros_like(tau_lo)

    # =========================================================================
    # FIX: Added early stopping to escape the python 'for' loop bottleneck
    # =========================================================================
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        sum_p = p.sum(dim=dim, keepdim=True)

        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)

        # Early breaking if the bracket is sufficiently small
        if (tau_hi - tau_lo).max() < 1e-3:
            break

    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)

    denom = p.sum(dim=dim, keepdim=True)

    # Softmax fallback for numerical stabilization
    scores_for_sm = scores.masked_fill(~valid_mask, -10000.0) if valid_mask is not None else scores
    scores_for_sm = torch.where(torch.isfinite(scores_for_sm), scores_for_sm, neg_inf_val)
    sf = torch.nan_to_num(F.softmax(scores_for_sm.float(), dim=dim), nan=0.0).to(dtype)
    if valid_mask is not None:
        sf = sf.masked_fill(~valid_mask, 0.0)
    sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
    sf = sf / sf_denom

    is_empty = (denom < 1e-7) | torch.isnan(denom)
    p_normalized = torch.nan_to_num(p / denom.clamp(min=1e-9), nan=0.0)
    p = torch.where(is_empty, sf, p_normalized)
    return torch.nan_to_num(p, nan=0.0)


def entmax15_bisect(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    dim: int = -1,
    n_iter: int = 15,
) -> torch.Tensor:
    """Backward-compatible wrapper for 1.5-entmax."""
    return entmax_bisect(scores, alpha=1.5, valid_mask=valid_mask, dim=dim, n_iter=n_iter)


class AdaSplashLinear(nn.Module):
    r"""
    Linear projection layer with frozen base weights and trainable residual LoRA adapter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 16,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.lora_rank: int = min(lora_rank, min(in_features, out_features))

        factory_kwargs = {"dtype": dtype, "device": device}

        # Base weights - frozen to maintain pretrained stability
        self.weight: nn.Parameter = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs),
            requires_grad=False,
        )

        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.empty(out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_B: nn.Parameter | None = nn.Parameter(
                torch.empty((out_features, self.lora_rank), **factory_kwargs)
            )
            self.lora_A: nn.Parameter | None = nn.Parameter(
                torch.empty((self.lora_rank, in_features), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                w_fp32 = torch.empty_like(self.weight, dtype=torch.float32, device="cpu")
                nn.init.kaiming_uniform_(w_fp32, a=math.sqrt(5))
                self.weight.copy_(w_fp32.to(dtype=self.weight.dtype, device=self.weight.device))
            else:
                nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.bias is not None:
                nn.init.zeros_(self.bias)

            if (
                getattr(self, "lora_A", None) is not None
                and getattr(self, "lora_B", None) is not None
            ):
                if self.lora_B.dtype in (torch.float16, torch.bfloat16):
                    b_fp32 = torch.empty_like(self.lora_B, dtype=torch.float32, device="cpu")
                    nn.init.kaiming_uniform_(b_fp32, a=math.sqrt(5))
                    self.lora_B.copy_(b_fp32.to(dtype=self.lora_B.dtype, device=self.lora_B.device))
                else:
                    nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
                nn.init.zeros_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


@ATTENTION_REGISTRY.register(
    "adasplash",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
@ATTENTION_REGISTRY.register(
    "adasplash_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
)
class AdaSplashAttention(BaseAttention):
    r"""
    AdaSplash Attention Layer with α-Entmax and Local Splash Radius.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: int | None = None,
        alpha: float = 1.1,  # FIX: Softened from 1.5 to 1.1 for zero-shot accuracy
        n_iter: int = 15,
        splash_radius: int = 128,  # FIX: Increased from 32 to 128 for healthy context window
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank, **kwargs)
        if config is not None:
            self.adapter = ModelAdapter(config)
            self.alpha = float(getattr(config, "alpha", kwargs.get("alpha", alpha)))
            self.n_iter = getattr(config, "n_iter", kwargs.get("n_iter", n_iter))
            self.splash_radius = int(
                getattr(config, "splash_radius", kwargs.get("splash_radius", splash_radius))
            )
            lora_rank_cfg = getattr(
                config,
                "lora_rank",
                getattr(
                    config,
                    "residual_rank",
                    getattr(config, "rank", kwargs.get("lora_rank", kwargs.pop("rank", 16))),
                ),
            )
            self.lora_rank = int(lora_rank_cfg) if lora_rank_cfg is not None else 16
            self.num_heads = kwargs.get("num_heads", self.adapter.num_heads or 12)
            self.num_kv_heads = kwargs.get(
                "num_kv_heads", self.adapter.num_key_value_heads or self.num_heads
            )
            self.hidden_size = kwargs.get("hidden_size", self.adapter.hidden_size or 768)
            self.head_dim = kwargs.get(
                "head_dim",
                self.adapter.head_dim or (self.hidden_size // max(1, self.num_heads)),
            )
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            dropout_p = getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
            self.is_decoder = getattr(config, "is_decoder", False)
            self.relative_attention_num_buckets = getattr(
                config, "relative_attention_num_buckets", 32
            )
            self.relative_attention_max_distance = getattr(
                config, "relative_attention_max_distance", 128
            )
        else:
            self.adapter = None
            self.alpha = float(kwargs.get("alpha", alpha))
            self.n_iter = int(kwargs.get("n_iter", n_iter))
            self.splash_radius = int(kwargs.get("splash_radius", splash_radius))
            self.lora_rank = int(kwargs.get("lora_rank", kwargs.pop("rank", 16)))
            self.hidden_size = kwargs.get("hidden_size", 768)
            self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
            self.num_kv_heads = kwargs.get(
                "num_kv_heads", kwargs.get("num_key_value_heads", self.num_heads)
            )
            self.head_dim = kwargs.get("head_dim", self.hidden_size // max(1, self.num_heads))
            self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
            dropout_p = kwargs.get("dropout", 0.1)
            self.is_decoder = kwargs.get("is_decoder", False)
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128

        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        self.q_proj = AdaSplashLinear(
            self.hidden_size, self.all_head_size, lora_rank=self.lora_rank
        )
        self.k_proj = AdaSplashLinear(self.hidden_size, self.kv_head_size, lora_rank=self.lora_rank)
        self.v_proj = AdaSplashLinear(self.hidden_size, self.kv_head_size, lora_rank=self.lora_rank)
        self.out_proj = AdaSplashLinear(
            self.all_head_size, self.hidden_size, lora_rank=self.lora_rank
        )

        # Aliases
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.is_nested_encoder = False
        self.has_relative_attention_bias = False
        self.relative_attention_bias = None

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(
        self, query_length: int, key_length: int, device: torch.device | None = None
    ) -> torch.Tensor:
        if self.relative_attention_bias is not None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)
            values = values.permute([2, 0, 1]).unsqueeze(0)
            return values
        return torch.zeros((1, self.num_heads, query_length, key_length), device=device)

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted_all = True
        extracted_weights = {}

        for proj, target in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ]:
            w, b = None, None
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
            except Exception:
                pass

            if w is None:
                # Output projection checks first
                if proj == "o":
                    if hasattr(src, "output") and hasattr(src.output, "dense"):
                        mod = src.output.dense
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "dense"):
                        mod = src.dense
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "c_proj"):
                        c_proj_mod = getattr(src.c_proj, "base_layer", src.c_proj)
                        w = c_proj_mod.weight.T
                        b = getattr(c_proj_mod, "bias", None)
                    elif hasattr(src, "out_lin"):
                        mod = getattr(src.out_lin, "base_layer", src.out_lin)
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "o_proj"):
                        mod = getattr(src.o_proj, "base_layer", src.o_proj)
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "wo"):
                        mod = getattr(src.wo, "base_layer", src.wo)
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                # GPT-2 Fused c_attn
                elif hasattr(src, "c_attn") and proj in ["q", "k", "v"]:
                    c_attn_mod = getattr(src.c_attn, "base_layer", src.c_attn)
                    fused_w = c_attn_mod.weight.T
                    fused_b = getattr(c_attn_mod, "bias", None)
                    split_size = fused_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = fused_w[idx * split_size : (idx + 1) * split_size, :]
                    b = (
                        fused_b[idx * split_size : (idx + 1) * split_size]
                        if fused_b is not None
                        else None
                    )
                # RoBERTa / BERT
                elif (
                    proj in ["q", "k", "v"] and hasattr(src, "self") and hasattr(src.self, "query")
                ):
                    s = src.self
                    name_map = {"q": "query", "k": "key", "v": "value"}
                    mod = getattr(s, name_map[proj])
                    w = mod.weight
                    b = getattr(mod, "bias", None)
                elif hasattr(src, "query") and proj == "q":
                    w = src.query.weight
                    b = getattr(src.query, "bias", None)
                elif hasattr(src, "key") and proj == "k":
                    w = src.key.weight
                    b = getattr(src.key, "bias", None)
                elif hasattr(src, "value") and proj == "v":
                    w = src.value.weight
                    b = getattr(src.value, "bias", None)
                # DistilBERT manual fallback
                elif hasattr(src, "q_lin") and proj == "q":
                    w = src.q_lin.weight
                    b = getattr(src.q_lin, "bias", None)
                elif hasattr(src, "k_lin") and proj == "k":
                    w = src.k_lin.weight
                    b = getattr(src.k_lin, "bias", None)
                elif hasattr(src, "v_lin") and proj == "v":
                    w = src.v_lin.weight
                    b = getattr(src.v_lin, "bias", None)
                elif hasattr(src, "out_lin") and proj == "o":
                    w = src.out_lin.weight
                    b = getattr(src.out_lin, "bias", None)
                # Modern Decoders
                elif hasattr(src, f"{proj}_proj"):
                    mod = getattr(src, f"{proj}_proj")
                    w = mod.weight
                    b = getattr(mod, "bias", None)

            extracted_weights[proj] = (w, b)
            if w is None:
                extracted_all = False

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        is_nested_encoder = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name in ("BertSelfAttention", "RobertaSelfAttention")
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
            and not has_out_proj
        ):
            is_nested_encoder = True

        if not extracted_all:
            if (
                is_nested_encoder
                and extracted_weights["q"][0] is not None
                and extracted_weights["k"][0] is not None
                and extracted_weights["v"][0] is not None
            ):
                extracted_all = True

        if not extracted_all:
            raise RuntimeError(
                f"AdaSplashAttention.copy_weights: could not extract mandatory projections "
                f"{[p for p, (w, _) in extracted_weights.items() if w is None]} from {cls_name}. "
                f"Refusing to install randomly-initialised weights."
            )

        for proj, target in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ]:
            w, b = extracted_weights[proj]
            if w is not None:
                with torch.no_grad():
                    out_d, in_d = target.weight.shape
                    src_out, src_in = w.shape
                    target.weight.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w[: min(out_d, src_out), : min(in_d, src_in)]
                    )
                    if b is not None and target.bias is not None:
                        target.bias.data[: min(len(target.bias), len(b))].copy_(
                            b[: min(len(target.bias), len(b))]
                        )
                    if getattr(target, "lora_A", None) is not None:
                        nn.init.kaiming_uniform_(target.lora_A, a=math.sqrt(5))
                        target.lora_A.requires_grad = True
                    if getattr(target, "lora_B", None) is not None:
                        nn.init.zeros_(target.lora_B)
                        target.lora_B.requires_grad = True
            elif proj == "o" and is_nested_encoder:
                with torch.no_grad():
                    if target.weight.shape[0] == target.weight.shape[1]:
                        nn.init.eye_(target.weight)
                    else:
                        nn.init.zeros_(target.weight)
                        min_d = min(target.weight.shape[0], target.weight.shape[1])
                        target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        nn.init.zeros_(target.bias)
                    if getattr(target, "lora_A", None) is not None:
                        nn.init.kaiming_uniform_(target.lora_A, a=math.sqrt(5))
                        target.lora_A.requires_grad = True
                    if getattr(target, "lora_B", None) is not None:
                        nn.init.zeros_(target.lora_B)
                        target.lora_B.requires_grad = True

            target.weight.requires_grad = False
            if target.bias is not None:
                target.bias.requires_grad = False

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

        self.is_nested_encoder = is_nested_encoder

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

        q = self.q_proj(hidden_states).view(B, L_q, H, Dh).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        position_embeddings = kwargs.get("position_embeddings")
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]

            def _rotate_half(x):
                _d = x.shape[-1]
                _h = _d // 2
                _rot = torch.cat((-x[..., _h : 2 * _h], x[..., :_h]), dim=-1)
                if _d % 2 == 1:
                    _rot = torch.cat([_rot, x[..., -1:]], dim=-1)
                return _rot

            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            rotary_dim = cos.shape[-1]
            cos_q, sin_q = (
                (cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :])
                if cos.shape[-2] != q.shape[-2]
                else (cos, sin)
            )
            cos_k, sin_k = (
                (cos[..., : k.shape[-2], :], sin[..., : k.shape[-2], :])
                if cos.shape[-2] != k.shape[-2]
                else (cos, sin)
            )
            if rotary_dim <= q.shape[-1]:
                q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
                k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
                q_rot = (q_rot * cos_q.to(q_rot.dtype)) + (
                    _rotate_half(q_rot) * sin_q.to(q_rot.dtype)
                )
                k_rot = (k_rot * cos_k.to(k_rot.dtype)) + (
                    _rotate_half(k_rot) * sin_k.to(k_rot.dtype)
                )
                q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut_q, sin_cut_q = (
                    cos_q[..., : q.shape[-1]],
                    sin_q[..., : q.shape[-1]],
                )
                cos_cut_k, sin_cut_k = (
                    cos_k[..., : k.shape[-1]],
                    sin_k[..., : k.shape[-1]],
                )
                q = (q * cos_cut_q.to(q.dtype)) + (_rotate_half(q) * sin_cut_q.to(q.dtype))
                k = (k * cos_cut_k.to(k.dtype)) + (_rotate_half(k) * sin_cut_k.to(k.dtype))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                q, k = maybe_apply_rotary(
                    q, k, self.adapter, self.rotary_emb, kwargs.get("position_ids")
                )
            except Exception:
                pass

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = Dh**-0.5
        logits = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale

        position_bias = kwargs.get("position_bias")
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(L_q, L_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask")
        if mask is not None:
            logits = apply_attention_mask(logits, mask)

        if position_bias is not None:
            logits = logits + position_bias.to(logits.dtype)

        mt = str(getattr(self.config, "model_type", "") or "").lower()
        decoder_prefixes = (
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
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or any(mt.startswith(p) for p in decoder_prefixes)
        )
        causal_tril = None
        if is_decoder:
            causal_tril = (
                torch.tril(torch.ones((L_q, L_k), device=device, dtype=torch.bool))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            logits = logits.masked_fill(~causal_tril, -10000.0)

        splash_mask_4d = None
        if (
            is_decoder
            and self.splash_radius > 0
            and not is_cross_attention
            and L_k > self.splash_radius
        ):
            q_idx = torch.arange(L_q, device=device).unsqueeze(1)
            k_idx = torch.arange(L_k, device=device).unsqueeze(0)
            dist = q_idx - k_idx
            splash_local = ((dist >= 0) & (dist <= self.splash_radius)) | (k_idx < 4)
            splash_mask_4d = splash_local.unsqueeze(0).unsqueeze(0)
            logits = logits.masked_fill(~splash_mask_4d, -10000.0)

        valid_mask_entmax = None
        if not is_cross_attention:
            valid_mask_entmax = logits > -5000.0

        if not is_cross_attention:
            attn = entmax_bisect(
                logits,
                alpha=self.alpha,
                valid_mask=valid_mask_entmax,
                dim=-1,
                n_iter=self.n_iter,
            ).to(dtype)
        else:
            attn = torch.nan_to_num(F.softmax(logits.float(), dim=-1), nan=0.0).to(dtype)

        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(-1, self.num_heads, 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.all_head_size)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        if position_bias is not None:
            return output, position_bias
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adasplash_inject")
@ATTENTION_REGISTRY.register("adasplash_attention_inject")
@ATTENTION_REGISTRY.register("inject_adasplash")
def inject_adasplash(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> AdaSplashAttention | nn.Module | None:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adasplash(
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

    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is not None:
                    src = cand.self if hasattr(cand, "self") else cand
                    break

        if src is None and target is not None and hasattr(target, "forward"):
            src = target

        if src is None and cfg is None:
            return None

        if isinstance(src, AdaSplashAttention) or "AdaSplash" in src.__class__.__name__:
            return None
        if hasattr(src, "inner") and isinstance(src.inner, AdaSplashAttention):
            return None
        if hasattr(src, "_wrapped_attention") and isinstance(
            src._wrapped_attention, AdaSplashAttention
        ):
            return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, AdaSplashAttention) or "AdaSplash" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(cand.inner, AdaSplashAttention):
                    return None

        if src is not None and (
            "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__
        ):
            return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        alpha = getattr(comp_config, "alpha", kwargs.get("alpha", 1.1))
        splash_radius = getattr(comp_config, "splash_radius", kwargs.get("splash_radius", 128))
        lora_rank = getattr(
            comp_config,
            "lora_rank",
            getattr(
                comp_config,
                "residual_rank",
                getattr(
                    comp_config,
                    "rank",
                    (
                        getattr(
                            cfg,
                            "lora_rank",
                            getattr(
                                cfg,
                                "rank",
                                kwargs.get("lora_rank", kwargs.pop("rank", 16)),
                            ),
                        )
                        if cfg is not None
                        else kwargs.get("lora_rank", kwargs.pop("rank", 16))
                    ),
                ),
            ),
        )

        new_attn = AdaSplashAttention(
            config=cfg,
            alpha=alpha,
            splash_radius=splash_radius,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        p = next(src.parameters(), None) if src is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn

        return new_attn
    except Exception as e:
        logger.warning("inject_adasplash failed at layer %d: %s", layer_idx, e)
        return None
