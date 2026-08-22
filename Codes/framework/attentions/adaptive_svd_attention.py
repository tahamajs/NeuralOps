r"""
Adaptive SVD Attention with Residual Adapters.

This module provides low-rank factorized linear projections for Multi-Head Attention,
decomposing full-rank weight matrices via truncated Singular Value Decomposition (SVD)
augmented with trainable low-rank residual adapters:
$$W \approx U V + A B$$
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
        $$y = x V^T U^T + x B^T A^T + b$$
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
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.svd_rank: int = svd_rank
        self.residual_rank: int = residual_rank

        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, svd_rank))
        self.V: nn.Parameter = nn.Parameter(torch.empty(svd_rank, in_features))

        if residual_rank > 0:
            self.A: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features, residual_rank))
            self.B: Optional[nn.Parameter] = nn.Parameter(torch.empty(residual_rank, in_features))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Initializes parameters using orthogonal base and zeroed residual adapter (AB = 0 at init)."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.A is not None and self.B is not None:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        r"""Sets the SVD factors $U$ and $V$ with precomputed truncated SVD decomposition."""
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)
            if self.B is not None:
                nn.init.zeros_(self.B)

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
class AdaptiveSVDAttention(nn.Module):
    r"""Multi-Head Attention with Adaptive SVD + Residual factorized projections."""

    def __init__(
        self,
        config: Any,
        svd_rank: Optional[int] = None,
        residual_rank: int = 8,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = config.hidden_size
        self.svd_rank: int = getattr(
            config,
            "svd_rank",
            svd_rank if svd_rank is not None else getattr(config, "rank", 32),
        )
        self.residual_rank: int = getattr(config, "residual_rank", residual_rank)

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

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _init_svd_part(self, adap_linear: AdaptiveSVDLinear, W_data: torch.Tensor, bias_data: Optional[torch.Tensor]) -> None:
        with torch.no_grad():
            W = W_data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(adap_linear.svd_rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(S[:r])
            U_r = (U[:, :r] * sqrt_s).to(adap_linear.U.dtype)
            V_r = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(adap_linear.V.dtype)
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
            except Exception:
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Adaptive SVD Attention with universal masking."""
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.key(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.value(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = apply_attention_mask(scores, attention_mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None
