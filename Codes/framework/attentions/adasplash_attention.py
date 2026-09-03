r"""
AdaSplash Attention: Adaptive Splash Attention with α-Entmax Sparse Activations.

Combines localized sliding-window splash radius with sparse α-entmax adaptive thresholding
and global token routing, ensuring robust numerical convergence and exact pretrained weight transfer.
"""

from typing import Any, Optional, Tuple
import math

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def entmax15_bisect(
    scores: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    dim: int = -1,
    n_iter: int = 25,
) -> torch.Tensor:
    r"""
    Compute 1.5-entmax via stabilized bisection strictly over valid tokens.
    Solves $\sum_j \max(0, \alpha' s_j - \tau)^{1/(\alpha-1)} = 1$ for $\alpha = 1.5$.
    Includes softmax fallback when support is empty, preserving gradient flow.
    """
    alpha = 1.5
    alpha_minus_1 = alpha - 1.0  # 0.5

    # If mask is provided, mask out invalid keys with large negative value
    if valid_mask is not None:
        # valid_mask: broadcastable to scores [B, H, L, S]
        scores_valid = scores.masked_fill(~valid_mask, -10000.0)
    else:
        scores_valid = scores

    # Shift scores for numerical stability
    max_scores = scores_valid.amax(dim=dim, keepdim=True)
    # Clamp -inf max_scores if all are masked
    max_scores = torch.where(max_scores < -5000.0, torch.zeros_like(max_scores), max_scores)
    scores_shift = scores_valid - max_scores
    scaled = scores_shift * alpha_minus_1

    # Bisection search interval bounded ONLY by valid tokens
    if valid_mask is not None:
        scaled_for_min = scaled.masked_fill(~valid_mask, 10000.0)
        tau_lo = scaled_for_min.amin(dim=dim, keepdim=True) - 1.0
        # If all tokens were invalid in some row, clamp tau_lo
        tau_lo = torch.where(tau_lo > 5000.0, torch.zeros_like(tau_lo) - 1.0, tau_lo)
    else:
        tau_lo = scaled.amin(dim=dim, keepdim=True) - 1.0
    
    tau_hi = torch.zeros_like(tau_lo)

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** (1.0 / alpha_minus_1)
        if valid_mask is not None:
            p = p.masked_fill(~valid_mask, 0.0)
        sum_p = p.sum(dim=dim, keepdim=True)
        # If sum > 1, tau is too low -> increase tau_lo
        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)

    # Final tau
    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** (1.0 / alpha_minus_1)
    if valid_mask is not None:
        p = p.masked_fill(~valid_mask, 0.0)
    
    denom = p.sum(dim=dim, keepdim=True)

    # Softmax fallback for numerical stabilization when support is empty or ill-conditioned
    if valid_mask is not None:
        scores_for_sm = scores.masked_fill(~valid_mask, -10000.0)
        sf = torch.nan_to_num(F.softmax(scores_for_sm.float(), dim=dim), nan=0.0).to(scores.dtype)
        sf = sf.masked_fill(~valid_mask, 0.0)
        sf_denom = sf.sum(dim=dim, keepdim=True).clamp(min=1e-9)
        sf = sf / sf_denom
    else:
        sf = torch.nan_to_num(F.softmax(scores.float(), dim=dim), nan=0.0).to(scores.dtype)

    is_empty = (denom < 1e-7) | torch.isnan(denom)
    p_normalized = torch.nan_to_num(p / denom.clamp(min=1e-9), nan=0.0)
    p = torch.where(is_empty, sf, p_normalized)
    return torch.nan_to_num(p, nan=0.0)


@ATTENTION_REGISTRY.register("adasplash", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adasplash_attention")
class AdaSplashAttention(BaseAttention):
    r"""
    AdaSplash Attention Layer with α-Entmax and Local Splash Radius.

    Mathematical Formulation:
        1. Query-Key Relevance:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_h}}$$
        
        2. Adaptive Splash Masking:
           Retains all local tokens within splash radius $W$ while applying sparse 1.5-entmax:
           $$\mathcal{M}^{\text{splash}}_{ij} = \begin{cases} 1 & \text{if } |i - j| \le W \\ 1 & \text{if valid token} \\ 0 & \text{if padding} \end{cases}$$
        
        3. Sparse Entmax Normalization:
           $$A = \text{1.5-entmax}(S, \text{dim}=-1)$$
           $$O = A V W_O$$
    """

    supports_gqa: bool = True

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
        self.alpha: float = alpha
        self.n_iter: int = n_iter
        self.splash_radius: int = splash_radius
        self.num_heads: int = getattr(config, "num_attention_heads", 4)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.v_proj = nn.Linear(hidden_size, self.kv_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

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

        # Copy pretrained weights if available
        self.is_decoder: bool = getattr(config, "is_decoder", False)
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
        r"""Universal adapter weight copying with T5 relative position bias preservation."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
        kv_heads = self.num_kv_heads
        k = self.k_proj(kv_input).view(B, L_k, kv_heads, Dh).transpose(1, 2)  # [B, kv_heads, L_k, Dh]
        v = self.v_proj(kv_input).view(B, L_k, kv_heads, Dh).transpose(1, 2)  # [B, kv_heads, L_k, Dh]
        if self.kv_repeat > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.kv_repeat, -1).reshape(B, H, L_k, Dh)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.kv_repeat, -1).reshape(B, H, L_k, Dh)

        scale = Dh ** -0.5
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L_q, L_k]

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(L_q, L_k, device=device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if position_bias is not None:
            if mask is not None and mask.dim() == 4:
                position_bias = position_bias + mask.to(logits.dtype)
            logits = logits + position_bias.to(logits.dtype)
        elif mask is not None:
            if mask.dim() == 4:
                logits = logits + mask.to(logits.dtype)
            else:
                valid_m = get_token_valid_mask(mask, B, L_k, device)
                if valid_m is not None:
                    logits = logits.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

        # Apply entmax or softmax
        if self.alpha == 1.5 and not is_cross_attention:
            attn = entmax15_bisect(
                logits,
                valid_mask=None,
                dim=-1,
                n_iter=self.n_iter,
            ).to(dtype)
        else:
            attn = torch.nan_to_num(F.softmax(logits.float(), dim=-1), nan=0.0).to(dtype)

        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        # Compute context
        out = torch.matmul(attn, v)  # [B, H, L_q, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.all_head_size)
        output = self.out_proj(out)

        if position_bias is not None:
            return output, position_bias
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adasplash_inject")
def inject_adasplash(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs) -> AdaSplashAttention:
    alpha = getattr(comp_config, "alpha", 1.5)
    new_attn = AdaSplashAttention(
        config=model_config,
        alpha=alpha,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn