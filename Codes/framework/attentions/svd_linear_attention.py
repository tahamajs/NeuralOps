"""
SVD Linear Attention – each linear projection (Q, K, V, Out) is replaced by
a low‑rank factorisation: W = U @ V, computed as U(Vx) + bias.
Zero‑shot initialisation from any pretrained Transformer attention layer.
"""

import logging
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


class SVDLinear(nn.Module):
    r"""Low-Rank Factorized Linear Layer.

    Approximates a dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
    by factorizing into two low-rank matrices $U \in \mathbb{R}^{d_{\text{out}} \times r}$ and
    $V \in \mathbb{R}^{r \times d_{\text{in}}}$ where $r \ll \min(d_{\text{in}}, d_{\text{out}})$:
    $$y = x W^\top + b \approx x V^\top U^\top + b = (x V^\top) U^\top + b$$

    Computational Complexity:
        - Parameters: $r(d_{\text{in}} + d_{\text{out}})$ vs $d_{\text{in}} \cdot d_{\text{out}}$
        - FLOPs: $\mathcal{O}(B \cdot N \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$
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
        """Forward pass through factorized linear projections."""
        Vx: torch.Tensor = F.linear(x, self.V)  # (..., rank)
        out: torch.Tensor = F.linear(Vx, self.U)  # (..., out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def set_low_rank_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Explicitly set singular vectors into weight parameters."""
        with torch.no_grad():
            self.U.zero_()
            self.V.zero_()
            r_u = min(self.U.size(1), U.size(1))
            r_v = min(self.V.size(0), V.size(0))
            self.U[:, :r_u].copy_(U[:, :r_u])
            self.V[:r_v, :].copy_(V[:r_v, :])


@ATTENTION_REGISTRY.register("svd_linear_attn")
@ATTENTION_REGISTRY.register("svd_linear")
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
        config: Any,
        rank: Optional[int] = None,
        pretrained_attn: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = config.hidden_size
        self.rank: int = getattr(
            config,
            "svd_rank",
            rank if rank is not None else getattr(config, "rank", 32),
        )

        self.query: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.key: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.value: SVDLinear = SVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
        )
        self.out_proj: SVDLinear = SVDLinear(
            self.all_head_size,
            self.hidden_size,
            self.rank,
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
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _init_from_weight(
        self, svd_linear: SVDLinear, W_data: torch.Tensor, bias_data: Optional[torch.Tensor]
    ) -> None:
        """Initialize SVD factors safely from weight and bias."""
        with torch.no_grad():
            W = W_data.float()
            # Ensure zeroed initial state for truncated SVD columns
            svd_linear.U.data.zero_()
            svd_linear.V.data.zero_()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            svd_linear.U.data[:, :r] = (U[:, :r] * S[:r]).to(svd_linear.U.dtype)
            svd_linear.V.data[:r, :] = Vt[:r, :].to(svd_linear.V.dtype)
            if svd_linear.bias is not None and bias_data is not None:
                svd_linear.bias.data.copy_(bias_data)

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                self._init_from_weight(target, w, b)
            except Exception as e:
                logger.debug("SVDLinearAttention copy_weights skipped %s: %s", proj, e)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SVD factorized linear attention."""
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q: torch.Tensor = (
            self.query(hidden_states).reshape(B, N, H, D).transpose(1, 2)
        )
        K: torch.Tensor = (
            self.key(hidden_states).reshape(B, N, H, D).transpose(1, 2)
        )
        V: torch.Tensor = (
            self.value(hidden_states).reshape(B, N, H, D).transpose(1, 2)
        )

        scale = D**-0.5
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = apply_attention_mask(scores, attention_mask)

        attn_weights: torch.Tensor = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context: torch.Tensor = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().reshape(B, N, self.all_head_size)
        output: torch.Tensor = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)
