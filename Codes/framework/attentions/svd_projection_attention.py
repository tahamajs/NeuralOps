"""
SVD Projection Attention – Each projection (Q, K, V, Out) is replaced by
a low‑rank SVDLinear with explicit weight matrix reconstruction: W = U @ V.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


class SVDProjectionLinear(nn.Module):
    r"""Low-Rank Factorized Projection Layer with Dynamic Matrix Reconstruction.

    Stores orthogonal singular subspaces $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and
    $V \in \mathbb{R}^{r \times d_{\text{in}}}$, reconstructing the dense linear projection:
    $$W = U V \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$
    $$y = x W^\top + b = x (U V)^\top + b$$

    Computational Complexity:
        - Memory Footprint: $\mathcal{O}(r(d_{\text{in}} + d_{\text{out}}))$
        - Forward Time: $\mathcal{O}(B \cdot N \cdot d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}} \cdot r \cdot d_{\text{in}})$

    Tensor Dimensions:
        - Input $x$: $[*, d_{\text{in}}]$
        - Output $y$: $[*, d_{\text{out}}]$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, rank))
        self.V: nn.Parameter = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with Kaiming uniform distribution."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input using reconstructed low-rank weight.

        Args:
            x (torch.Tensor): Input activations of shape `[*, in_features]`.

        Returns:
            torch.Tensor: Output activations of shape `[*, out_features]`.
        """
        weight: torch.Tensor = self.U @ self.V
        return F.linear(x, weight, self.bias)


class SVDProjectionAttention(nn.Module):
    r"""Multi-Head Attention using Reconstructed SVD Linear Projections.

    Constructs dense projection operators from low-rank singular subspaces with zero-shot
    initialization from pretrained checkpoints:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(X (U_q V_q)^\top)(X (U_k V_k)^\top)^\top}{\sqrt{d_k}} + M\right) (X (U_v V_v)^\top)$$

    Computational Complexity:
        - Parameter Compression: $4 \cdot r(d + d_{\text{all}})$ vs $4 \cdot d \cdot d_{\text{all}}$
        - Attention: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$
    """

    def __init__(
        self,
        model_config: Any,
        rank: int = 32,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads: int = model_config.num_attention_heads
        self.head_dim: int = model_config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = model_config.hidden_size
        self.rank: int = rank

        self.q_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.q_lin.bias is not None,
        )
        self.k_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.k_lin.bias is not None,
        )
        self.v_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.v_lin.bias is not None,
        )
        self.out_proj: SVDProjectionLinear = SVDProjectionLinear(
            self.all_head_size,
            self.hidden_size,
            rank,
            bias=pretrained_attn.out_lin.bias is not None,
        )

        self._init_from_linear(self.q_proj, pretrained_attn.q_lin)
        self._init_from_linear(self.k_proj, pretrained_attn.k_lin)
        self._init_from_linear(self.v_proj, pretrained_attn.v_lin)
        self._init_from_linear(self.out_proj, pretrained_attn.out_lin)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def _init_from_linear(
        self, svd_linear: SVDProjectionLinear, orig_linear: nn.Linear
    ) -> None:
        """ init from linear."""
        with torch.no_grad():
            W = orig_linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            svd_linear.U.data[:, :r] = (U[:, :r] * S[:r]).to(svd_linear.U.dtype)
            svd_linear.V.data[:r, :] = Vt[:r, :].to(svd_linear.V.dtype)
            if svd_linear.bias is not None and orig_linear.bias is not None:
                svd_linear.bias.data = orig_linear.bias.data.clone()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SVD projection attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output features and optional attention weights.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q: torch.Tensor = (
            self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )
        K: torch.Tensor = (
            self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )
        V: torch.Tensor = (
            self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        )

        scale = D**-0.5
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -10000.0

        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context: torch.Tensor = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output: torch.Tensor = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("svd_proj_attn")
def inject_svd_proj_attn(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SVDProjectionAttention:
    """Inject SVD Projection Attention into target model layer."""
    rank: int = getattr(comp_config, "svd_rank", 32)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn: SVDProjectionAttention = SVDProjectionAttention(
        model_config, rank, dropout, old_attn
    )
    layer.attention = new_attn
    return new_attn
