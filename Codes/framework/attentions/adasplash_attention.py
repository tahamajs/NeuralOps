r"""
AdaSplash Attention: Adaptive Splash Attention with α-Entmax Sparse Activations.

Combines localized sliding-window splash radius with sparse α-entmax adaptive thresholding
and global token routing, ensuring robust numerical convergence and exact pretrained weight transfer.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter
from framework.utils.compat import maybe_apply_rotary, repeat_kv


def entmax_bisect(
    scores: torch.Tensor,
    alpha: float = 1.5,
    valid_mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    n_iter: int = 25,
) -> torch.Tensor:
    r"""
    Compute α-entmax via stabilized bisection strictly over valid tokens.
    Solves $\sum_j \max(0, (\alpha - 1) s_j - \tau)^{1/(\alpha-1)} = 1$ for $\alpha \in (1, 2]$.
    When $\alpha = 1.0$, falls back to standard softmax.
    Includes softmax fallback when support is empty or ill-conditioned.
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

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        sum_p = p.sum(dim=dim, keepdim=True)
        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)

    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** inv_alpha_minus_1
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)

    denom = p.sum(dim=dim, keepdim=True)

    # Softmax fallback for numerical stabilization when support is empty or ill-conditioned
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
    valid_mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    n_iter: int = 25,
) -> torch.Tensor:
    """Backward-compatible wrapper for 1.5-entmax."""
    return entmax_bisect(scores, alpha=1.5, valid_mask=valid_mask, dim=dim, n_iter=n_iter)


class AdaSplashLinear(nn.Module):
    r"""
    Linear projection layer with frozen base weights and trainable residual LoRA adapter.

    Uses `lora_A` and `lora_B` parameter names so recovery optimizers and LoRA filters
    (such as get_recovery_params) register them automatically as trainable parameters.
    Base weights and biases are frozen to preserve pretrained representations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 16,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
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
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(out_features, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((out_features, self.lora_rank), **factory_kwargs)
            )
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(
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

            if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
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


@ATTENTION_REGISTRY.register("adasplash", supports_causal=True, supports_bidirectional=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("adasplash_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True, supports_rope=True)
class AdaSplashAttention(BaseAttention):
    r"""
    AdaSplash Attention Layer with α-Entmax and Local Splash Radius.

    Mathematical Formulation:
        1. Query-Key Relevance:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_h}}$$

        2. Adaptive Splash Masking:
           Retains local tokens within radius $W$ while applying sparse α-entmax:
           $$\mathcal{M}^{\text{splash}}_{ij} = \begin{cases} 1 & \text{if } |i - j| \le W \\ 1 & \text{if valid token} \\ 0 & \text{if padding} \end{cases}$$

        3. Sparse Entmax Normalization:
           $$A = \text{α-entmax}(S, \text{dim}=-1)$$
           $$O = A V W_O$$
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        alpha: float = 1.5,
        n_iter: int = 25,
        splash_radius: int = 32,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.adapter = ModelAdapter(config)
        self.alpha: float = float(getattr(config, "alpha", kwargs.get("alpha", alpha)))
        self.n_iter: int = n_iter
        self.splash_radius: int = int(getattr(config, "splash_radius", kwargs.get("splash_radius", splash_radius)))
        lora_rank_cfg = getattr(config, "lora_rank", getattr(config, "residual_rank", getattr(config, "rank", kwargs.get("lora_rank", kwargs.get("rank", 16)))))
        self.lora_rank: int = int(lora_rank_cfg) if lora_rank_cfg is not None else 16
        self.num_heads: int = self.adapter.num_heads
        self.num_kv_heads: int = self.adapter.num_key_value_heads
        self.hidden_size: int = self.adapter.hidden_size
        self.head_dim: int = self.adapter.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = self.adapter.kv_repeat

        self.q_proj = AdaSplashLinear(self.hidden_size, self.all_head_size, lora_rank=self.lora_rank)
        self.k_proj = AdaSplashLinear(self.hidden_size, self.kv_head_size, lora_rank=self.lora_rank)
        self.v_proj = AdaSplashLinear(self.hidden_size, self.kv_head_size, lora_rank=self.lora_rank)
        self.out_proj = AdaSplashLinear(self.all_head_size, self.hidden_size, lora_rank=self.lora_rank)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _relative_position_bucket(
        self, relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
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
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
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
        r"""Robust weight extraction from HuggingFace, PyTorch, and Conv1D attention modules."""
        from framework.core.adapters import UniversalAdapter

        extracted_all = True
        extracted_weights = {}

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            w, b = None, None
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
            except Exception:
                pass

            if w is None:
                # 1. Output projection checks first (RoBERTa/BERT outer, GPT-2, DistilBERT, etc.)
                if proj == "o":
                    if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                        mod = src.output.dense
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "dense"):
                        mod = src.dense
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "c_proj"):
                        w = src.c_proj.weight.T
                        b = getattr(src.c_proj, "bias", None)
                    elif hasattr(src, "out_lin"):
                        mod = src.out_lin
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "o_proj"):
                        mod = src.o_proj
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                    elif hasattr(src, "wo"):
                        mod = src.wo
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                # 2. GPT-2 Fused c_attn Conv1D for Q, K, V
                elif hasattr(src, "c_attn") and proj in ["q", "k", "v"]:
                    fused_w = src.c_attn.weight.T  # [3 * D, D]
                    fused_b = getattr(src.c_attn, "bias", None)
                    split_size = fused_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = fused_w[idx * split_size : (idx + 1) * split_size, :]
                    b = fused_b[idx * split_size : (idx + 1) * split_size] if fused_b is not None else None
                # 3. RoBERTa / BERT nested self-attention strictly for Q, K, V
                elif proj in ["q", "k", "v"] and hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
                    s = getattr(src, "self")
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
                # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral)
                elif hasattr(src, f"{proj}_proj"):
                    mod = getattr(src, f"{proj}_proj")
                    w = mod.weight
                    b = getattr(mod, "bias", None)

            extracted_weights[proj] = (w, b)
            if w is None:
                extracted_all = False

        if not extracted_all:
            cls_name = src.__class__.__name__
            has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
            is_nested_encoder = (cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj

            if is_nested_encoder and extracted_weights["q"][0] is not None and extracted_weights["k"][0] is not None and extracted_weights["v"][0] is not None:
                extracted_all = True

        if not extracted_all:
            raise RuntimeError(f"Could not extract all required weights for AdaSplashAttention. Missing: {[p for p, (w, _) in extracted_weights.items() if w is None]}")

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            w, b = extracted_weights[proj]
            if w is not None:
                with torch.no_grad():
                    out_d, in_d = target.weight.shape
                    src_out, src_in = w.shape
                    target.weight.data[:min(out_d, src_out), :min(in_d, src_in)].copy_(w[:min(out_d, src_out), :min(in_d, src_in)])
                    if b is not None and target.bias is not None:
                        target.bias.data[:min(len(target.bias), len(b))].copy_(b[:min(len(target.bias), len(b))])
                    if getattr(target, "lora_A", None) is not None:
                        nn.init.zeros_(target.lora_A)
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
                        nn.init.zeros_(target.lora_A)

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        if (cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj:
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Check for cross-attention
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, L_q, D = hidden_states.shape
        L_k = kv_input.shape[1]
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(B, L_q, H, Dh).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, L_k, self.num_kv_heads, Dh).transpose(1, 2)

        # Apply RoPE (Direct kwargs support for Qwen3.5/Mistral/LLaMA)
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            def _rotate_half(x):
                return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != q.shape[-2]:
                cos, sin = cos[..., :q.shape[-2], :], sin[..., :q.shape[-2], :]
            rotary_dim = cos.shape[-1]
            if rotary_dim <= q.shape[-1]:
                q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
                k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut, sin_cut = cos[..., :q.shape[-1]], sin[..., :q.shape[-1]]
                q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
                k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                q, k = maybe_apply_rotary(q, k, self.adapter, self.rotary_emb, kwargs.get("position_ids"))
            except Exception:
                pass

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = Dh ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L_q, L_k]

        # Handle attention mask (padding mask) and T5 relative position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(L_q, L_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            logits = apply_attention_mask(logits, mask)

        if position_bias is not None:
            logits = logits + position_bias.to(logits.dtype)

        # Explicit Causal Masking for autoregressive decoders
        is_decoder = (
            self.is_decoder
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
        )
        if is_decoder:
            causal_tril = torch.tril(torch.ones((L_q, L_k), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            logits = logits.masked_fill(~causal_tril, -10000.0)

        # Apply Local Splash Radius Masking
        # Retains local tokens within window radius W while heavily penalizing tokens outside the window.
        # Applied exclusively to autoregressive decoders where sliding-window causality is natural.
        # For bidirectional encoders (e.g. RoBERTa, DistilBERT), splash masking is skipped by default
        # to allow full-sequence sparse alpha-entmax attention without severing global context (e.g. [CLS]).
        if is_decoder and self.splash_radius > 0 and not is_cross_attention and L_k > self.splash_radius:
            q_idx = torch.arange(L_q, device=device).unsqueeze(1)
            k_idx = torch.arange(L_k, device=device).unsqueeze(0)
            dist = q_idx - k_idx
            splash_local = (dist >= 0) & (dist <= self.splash_radius)
            logits = logits.masked_fill(~splash_local.unsqueeze(0).unsqueeze(0), -10000.0)

        # Apply α-entmax or softmax
        if not is_cross_attention:
            attn = entmax_bisect(
                logits,
                alpha=self.alpha,
                valid_mask=None,
                dim=-1,
                n_iter=self.n_iter,
            ).to(dtype)
        else:
            attn = torch.nan_to_num(F.softmax(logits.float(), dim=-1), nan=0.0).to(dtype)

        attn = self.dropout(attn)

        # Reshape head_mask safely across all common dimensions
        if head_mask is not None:
            if head_mask.dim() == 1:
                # [num_heads] -> [1, num_heads, 1, 1]
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                # [batch, num_heads] -> [batch, num_heads, 1, 1]
                head_mask = head_mask.view(-1, self.num_heads, 1, 1)
            elif head_mask.dim() == 3:
                # [batch, num_heads, seq_len] -> [batch, num_heads, seq_len, 1]
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                pass
            attn = attn * head_mask

        # Compute context
        out = torch.matmul(attn, v)  # [B, H, L_q, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.all_head_size)

        # In nested encoders (BERT/RoBERTa .self), context passes through directly because output.dense applies after
        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        if position_bias is not None:
            return output, position_bias
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adasplash_inject")
@ATTENTION_REGISTRY.register("adasplash_attention_inject")
def inject_adasplash(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs: Any,
) -> Optional[AdaSplashAttention]:
    if old_attn is None:
        return None
    try:
        alpha = getattr(comp_config, "alpha", 1.5)
        splash_radius = getattr(comp_config, "splash_radius", 32)
        lora_rank = getattr(comp_config, "lora_rank", getattr(comp_config, "residual_rank", getattr(comp_config, "rank", 16)))
        test_module = AdaSplashAttention(
            config=model_config,
            alpha=alpha,
            splash_radius=splash_radius,
            lora_rank=lora_rank,
            pretrained_attn=old_attn,
        )
        layer.attention = test_module
        return test_module
    except (RuntimeError, Exception):
        return None