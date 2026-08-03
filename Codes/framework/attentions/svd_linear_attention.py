"""
SVD Linear Attention – each linear projection (Q, K, V, Out) is replaced by
a low‑rank factorisation: W = U @ V, computed as U(Vx) + bias.
Zero‑shot initialisation from a pretrained DistilBERT attention layer.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


class SVDLinear(nn.Module):
    r"""Low-Rank Factorized Linear Layer.

    Approximates a dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
    by factorizing into two low-rank matrices $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and
    $V \in \mathbb{R}^{r \times d_{\text{in}}}$ where $r \ll \min(d_{\text{in}}, d_{\text{out}})$:
    $$y = x W^\top + b \approx x V^\top U^\top + b = (x V^\top) U^\top + b$$

    Computational Complexity:
        - Parameters: $r(d_{\text{in}} + d_{\text{out}})$ vs $d_{\text{in}} \cdot d_{\text{out}}$
        - FLOPs: $\mathcal{O}(B \cdot N \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$

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
        """Initialize projection matrices with semi-orthogonal weights."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through factorized linear projections.

        Args:
            x (torch.Tensor): Input activations of shape `[*, in_features]`.

        Returns:
            torch.Tensor: Projected tensor of shape `[*, out_features]`.
        """
        Vx: torch.Tensor = F.linear(x, self.V)  # (..., rank)
        out: torch.Tensor = F.linear(Vx, self.U)  # (..., out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def set_low_rank_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Explicitly set singular vectors into weight parameters."""
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)


class SVDLinearAttention(nn.Module):
    r"""Multi-Head Attention with Low-Rank SVD-Factorized Linear Projections.

    Every projection $W_q, W_k, W_v, W_o$ is factorized into $U \Sigma V^\top$ via truncated
    Singular Value Decomposition from pretrained model checkpoints, achieving zero-shot parameter compression:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(X V_q^\top U_q^\top)(X V_k^\top U_k^\top)^\top}{\sqrt{d_k}} + M\right) (X V_v^\top U_v^\top)$$

    Computational Complexity:
        - Linear Projections: $\mathcal{O}(B \cdot N \cdot r \cdot (d + d_{\text{all}}))$
        - Attention Map: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
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

        self.q_proj: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.q_lin.bias is not None,
        )
        self.k_proj: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.k_lin.bias is not None,
        )
        self.v_proj: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            rank,
            bias=pretrained_attn.v_lin.bias is not None,
        )
        self.out_proj: SVDLinear = SVDLinear(
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
        self, svd_linear: SVDLinear, orig_linear: nn.Linear
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
        """Forward pass of SVD factorized linear attention.

        Args:
            hidden_states (torch.Tensor): Input activations `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Attention mask `[Batch, SeqLen]` or `[Batch, 1, 1, SeqLen]`.
            head_mask (Optional[torch.Tensor]): Mask for individual attention heads.
            output_attentions (bool): Whether to return raw attention weight matrices.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation and optional attention weights.
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
            scores = scores + (1.0 - mask) * -1e9

        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context: torch.Tensor = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output: torch.Tensor = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("svd_linear_attn")
def inject_svd_linear_attn(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SVDLinearAttention:
    """Inject SVD Linear Attention into target model layer."""
    rank: int = getattr(comp_config, "svd_rank", 32)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn: SVDLinearAttention = SVDLinearAttention(
        model_config, rank, dropout, old_attn
    )
    layer.attention = new_attn
    return new_attn
