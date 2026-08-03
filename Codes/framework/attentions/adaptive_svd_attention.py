r"""
Adaptive SVD Attention with Residual Adapters.

This module provides low-rank factorized linear projections for Multi-Head Attention,
decomposing full-rank weight matrices via truncated Singular Value Decomposition (SVD)
augmented with trainable low-rank residual adapters:
$$W \approx U V + A B$$
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    Tensor Dimensions:
        - Input $x$: $[B, N, d_{\text{in}}]$ (or $[*, d_{\text{in}}]$).
        - Intermediate $x V^T$: $[B, N, r_{\text{svd}}]$.
        - Intermediate $x B^T$: $[B, N, r_{\text{res}}]$.
        - Output $y$: $[B, N, d_{\text{out}}]$.
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
        r"""Initializes parameters using orthogonal and Kaiming uniform distributions."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.A is not None and self.B is not None:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        r"""
        Sets the SVD factors $U$ and $V$ with precomputed truncated SVD decomposition.

        Args:
            U (torch.Tensor): Left singular matrix multiplied by singular values, shape $[d_{\text{out}}, r_{\text{svd}}]$.
            V (torch.Tensor): Right singular matrix transpose, shape $[r_{\text{svd}}, d_{\text{in}}]$.
        """
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward computation for low-rank SVD + residual linear layer.

        Args:
            x (torch.Tensor): Input activations of shape $[*, d_{\text{in}}]$.

        Returns:
            torch.Tensor: Output activations of shape $[*, d_{\text{out}}]$.
        """
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.A is not None and self.B is not None:
            Bx = F.linear(x, self.B)
            out = out + F.linear(Bx, self.A)
        if self.bias is not None:
            out = out + self.bias
        return out


class AdaptiveSVDAttention(nn.Module):
    r"""
    Multi-Head Attention with Adaptive SVD + Residual factorized projections.

    Decomposes the projection matrices $W_Q, W_K, W_V, W_O$ into:
    $$W \approx U V + A B$$
    where $U V$ is initialized from the top-$r_{\text{svd}}$ singular vectors of the pretrained weights,
    and $A B$ provides dynamic adaptation capacity of rank $r_{\text{res}}$.

    Complexity:
        - Projections: $\mathcal{O}(B \cdot N \cdot D \cdot (r_{\text{svd}} + r_{\text{res}}))$ FLOPs.
        - Attention Logits & Softmax: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_h)$ where $d_h = D / H$.
        - Total Space: $\mathcal{O}(B \cdot H \cdot N^2)$ peak activation memory for attention maps.

    Tensor Dimensions:
        - `hidden_states`: $[B, N, D]$
        - Projected $Q, K, V$: $[B, H, N, d_h]$
        - Attention Scores / Weights: $[B, H, N, N]$
        - Context: $[B, N, D]$
        - Output: $[B, N, D]$
    """

    def __init__(
        self,
        model_config: Any,
        svd_rank: int = 32,
        residual_rank: int = 8,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot initialization")

        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size

        self.q_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", None)).bias is not None,
        )
        self.k_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", None)).bias is not None,
        )
        self.v_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", None)).bias is not None,
        )
        self.out_proj = AdaptiveSVDLinear(
            self.all_head_size,
            self.hidden_size,
            svd_rank,
            residual_rank,
            bias=getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None)).bias is not None,
        )

        # Initialize SVD part from pretrained weights
        q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", None))
        k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", None))
        v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", None))
        out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", None))

        self._init_svd_part(self.q_proj, q_src)
        self._init_svd_part(self.k_proj, k_src)
        self._init_svd_part(self.v_proj, v_src)
        self._init_svd_part(self.out_proj, out_src)

        self.dropout = nn.Dropout(dropout)

    def _init_svd_part(self, adap_linear: AdaptiveSVDLinear, orig_linear: nn.Linear) -> None:
        r"""
        Initializes $U$ and $V$ via truncated SVD of the original linear layer weights.

        Args:
            adap_linear (AdaptiveSVDLinear): Target low-rank linear layer.
            orig_linear (nn.Linear): Source pretrained full-rank linear module.
        """
        with torch.no_grad():
            W = orig_linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(adap_linear.svd_rank, U.size(1), Vt.size(0))
            U_r = (U[:, :r] * S[:r]).to(adap_linear.U.dtype)
            V_r = Vt[:r, :].to(adap_linear.V.dtype)
            adap_linear.set_svd_factors(U_r, V_r)
            if adap_linear.bias is not None and orig_linear.bias is not None:
                adap_linear.bias.data = orig_linear.bias.data.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass of Adaptive SVD Attention.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, N, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, N]$ or $[B, 1, 1, N]$.
            head_mask (Optional[torch.Tensor]): Head mask tensor of shape $[H]$ or $[B, H, N, N]$.
            output_attentions (bool): Whether to return the attention probability matrix.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context tensor of shape $[B, N, D]$.
                - Attention weights $[B, H, N, N]$ if `output_attentions=True`, else None.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            mask_float = mask.float()
            scores = scores + (1.0 - mask_float) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("adaptive_svd_attn")
def inject_adaptive_svd_attn(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AdaptiveSVDAttention:
    r"""
    Injects AdaptiveSVDAttention into a target Transformer layer.

    Args:
        layer (nn.Module): Transformer layer to inject attention into.
        model_config (Any): Architecture configuration.
        comp_config (Any): Compression configuration specifying `svd_rank`, `residual_rank`, and `dropout`.
        layer_idx (int): Layer index.
        total_layers (int): Total number of layers.
        old_attn (Optional[nn.Module]): Pretrained attention module to factorize.

    Returns:
        AdaptiveSVDAttention: Injected attention module.
    """
    svd_rank = getattr(comp_config, "svd_rank", 32)
    res_rank = getattr(comp_config, "residual_rank", 8)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = AdaptiveSVDAttention(model_config, svd_rank, res_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
