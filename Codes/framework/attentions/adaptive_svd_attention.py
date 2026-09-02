r"""
Adaptive SVD Attention with Residual Adapters and Dynamic Energy Thresholding.

This module provides low-rank factorized linear projections for Multi-Head Attention,
decomposing full-rank weight matrices via dynamic cumulative energy-thresholded
Singular Value Decomposition (SVD):
$$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
with symmetrical $\sqrt{\Sigma}$ singular value distribution:
$$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = \sqrt{\Sigma_r} V^\top_{:r, :}$$
augmented with trainable low-rank residual adapters:
$$W \approx U_r V_r + A B$$
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY


class AdaptiveSVDLinear(nn.Module):
    r"""
    Linear layer decomposed into a fixed/pretrained SVD base and a trainable residual adapter.

    Mathematical Formulation:
        Given an input $x \in \mathbb{R}^{B \times N \times d_{\text{in}}}$, the output is:
        $$y = x V^\top U^\top + x B^\top A^\top + b$$
        where:
        - $U \in \mathbb{R}^{d_{\text{out}} \times r_{\text{svd}}}$ and $V \in \mathbb{R}^{r_{\text{svd}} \times d_{\text{in}}}$
          represent the principal truncated SVD components ($W_{\text{svd}} = U V$).
        - $A \in \mathbb{R}^{d_{\text{out}} \times r_{\text{res}}}$ and $B \in \mathbb{R}^{r_{\text{res}} \times d_{\text{in}}}$
          represent the learnable residual adapter rank ($W_{\text{res}} = A B$).
        - $b \in \mathbb{R}^{d_{\text{out}}}$ is an optional bias vector.

    Complexity:
        - Computation: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r_{\text{svd}} + r_{\text{svd}} d_{\text{out}} + d_{\text{in}} r_{\text{res}} + r_{\text{res}} d_{\text{out}}))$ FLOPs.
        - Parameter Count: $(d_{\text{out}} + d_{\text{in}}) \cdot (r_{\text{svd}} + r_{\text{res}}) + d_{\text{out}}$.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        residual_rank: int = 8,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.svd_rank: int = svd_rank
        self.residual_rank: int = residual_rank

        factory_kwargs = {"dtype": dtype, "device": device}

        self.U: nn.Parameter = nn.Parameter(torch.empty((out_features, svd_rank), **factory_kwargs))
        self.V: nn.Parameter = nn.Parameter(torch.empty((svd_rank, in_features), **factory_kwargs))

        if residual_rank > 0:
            self.A: Optional[nn.Parameter] = nn.Parameter(torch.empty((out_features, residual_rank), **factory_kwargs))
            self.B: Optional[nn.Parameter] = nn.Parameter(torch.empty((residual_rank, in_features), **factory_kwargs))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Initializes parameters using orthogonal base and zeroed residual adapter (AB = 0 at init)."""
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if self.A is not None and self.B is not None:
                if self.A.dtype in (torch.float16, torch.bfloat16):
                    a_fp32 = torch.empty_like(self.A, dtype=torch.float32, device="cpu")
                    nn.init.kaiming_uniform_(a_fp32, a=math.sqrt(5))
                    self.A.copy_(a_fp32.to(dtype=self.A.dtype, device=self.A.device))
                else:
                    nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                nn.init.zeros_(self.B)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        r"""Sets the SVD factors $U$ and $V$ with precomputed truncated SVD decomposition."""
        with torch.no_grad():
            r = min(U.size(1), V.size(0), self.svd_rank)
            self.U.zero_()
            self.V.zero_()
            self.U[:, :r].copy_(U[:, :r])
            self.V[:r, :].copy_(V[:r, :])
            if self.B is not None:
                nn.init.zeros_(self.B)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W = torch.matmul(self.U, self.V)
        if self.A is not None and self.B is not None:
            W = W + torch.matmul(self.A, self.B)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward computation for low-rank SVD + residual linear layer."""
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.A is not None and self.B is not None:
            Bx = F.linear(x, self.B)
            out = out + F.linear(Bx, self.A)
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("adaptive_svd_attn")
@ATTENTION_REGISTRY.register("adaptive_svd")
@ATTENTION_REGISTRY.register("adaptive_svd_attention")
class AdaptiveSVDAttention(nn.Module):
    r"""Multi-Head Attention with Adaptive SVD + Residual factorized projections."""

    def __init__(
        self,
        config: Any,
        svd_rank: Optional[int] = None,
        residual_rank: int = 8,
        energy_threshold: float = 0.92,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = getattr(config, "num_attention_heads", getattr(config, "num_heads", 4))
        self.hidden_size: int = getattr(config, "hidden_size", getattr(config, "d_model", 768))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.svd_rank: int = getattr(
            config,
            "svd_rank",
            svd_rank if svd_rank is not None else getattr(config, "rank", 32),
        )
        self.residual_rank: int = getattr(config, "residual_rank", residual_rank)
        self.energy_threshold: float = float(
            getattr(
                config,
                "energy_threshold",
                kwargs.get("energy_threshold", energy_threshold),
            )
        )

        self.query = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.key = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.value = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.out_proj = AdaptiveSVDLinear(
            self.all_head_size,
            self.hidden_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
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

    def _compute_energy_rank(self, S: torch.Tensor, max_rank: int, min_rank: int = 16) -> int:
        r"""
        Find smallest rank $r$ satisfying cumulative singular value energy threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        """
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            return max(1, min(min_rank, max_rank))

        cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, min_rank), max_rank))

    def _init_svd_part(
        self,
        adap_linear: AdaptiveSVDLinear,
        W_data: torch.Tensor,
        bias_data: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            W = W_data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            max_r = min(adap_linear.svd_rank, U.size(1), Vt.size(0))
            r = self._compute_energy_rank(S, max_rank=max_r)

            # Symmetrical square-root singular value distribution:
            # U_r = U[:, :r] * sqrt(Sigma_r), V_r = sqrt(Sigma_r) * V_r^T
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).to(dtype=adap_linear.U.dtype, device=adap_linear.U.device)
            V_r = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(dtype=adap_linear.V.dtype, device=adap_linear.V.device)

            adap_linear.set_svd_factors(U_r, V_r)
            if adap_linear.bias is not None and bias_data is not None:
                adap_linear.bias.data.copy_(bias_data)

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                self._init_svd_part(target, w, b)
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
        """Forward pass of Adaptive SVD Attention with universal T5 & masking support."""
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N_q, H, D).transpose(1, 2)
        K = self.key(kv_input).view(B, N_k, H, D).transpose(1, 2)
        V = self.value(kv_input).view(B, N_k, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=hidden_states.device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if position_bias is not None:
            if mask is not None and mask.dim() == 4:
                position_bias = position_bias + mask.to(scores.dtype)
            scores = scores + position_bias.to(scores.dtype)
        elif mask is not None:
            scores = apply_attention_mask(scores, mask)

        attn_weights = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)
        output = self.out_proj(context)

        if position_bias is not None:
            return output, position_bias
        return output, attn_weights if output_attentions else None

