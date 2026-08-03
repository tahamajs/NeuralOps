import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class FactorisedLinear(nn.Module):
    r"""
    SVD-Factorized Linear Layer: $W \approx U \text{diag}(S) V^\top \to A B$.

    Mathematical Mechanism:
        $$A = U_{:, :r} \text{diag}(S_{:r}) \in \mathbb{R}^{\text{out} \times r}, \quad B = V_{:r, :} \in \mathbb{R}^{r \times \text{in}}$$
        $$y = (x B^\top) A^\top + b$$
    """

    def __init__(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_bias: Optional[torch.Tensor],
        energy: float = 0.95,
    ) -> None:
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total_energy = (S**2).sum()
            cum_energy = torch.cumsum(S**2, dim=0) / total_energy
            rank = int(torch.searchsorted(cum_energy, energy).item()) + 1
            rank = max(1, min(rank, min(out_dim, in_dim), 64))
            self.rank: int = rank

            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]

        self.down: nn.Linear = nn.Linear(in_dim, rank, bias=False)
        self.up: nn.Linear = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))

        self.down.weight.data = B.float()
        self.up.weight.data = A.float()

        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            self.up.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Evaluates factorized linear transformation."""
        return self.up(self.down(x))


@ATTENTION_REGISTRY.register("minimal_svd")
class MinimalSVDAttention(BaseAttention):
    r"""
    MinimalSVD: SVD-Factorized Low-Rank Projection Attention for DistilBERT.

    Mathematical Mechanism:
        Compresses attention projections using truncated singular value factorizations
        while maintaining full quadratic attention:
        $$\begin{aligned}
        Q &= \text{FactorisedLinear}_Q(X) \\
        K &= \text{FactorisedLinear}_K(X) \\
        V &= \text{FactorisedLinear}_V(X) \\
        S &= \frac{Q K^\top}{\sqrt{d_k}} + (1 - M) \cdot (-10^9) \\
        A &= \text{softmax}(S, \text{dim}=-1) \\
        O &= \text{FactorisedLinear}_O(A V)
        \end{aligned}$$

    Computational Complexity:
        - Projection Time: $\mathcal{O}(B \cdot N \cdot r \cdot D)$
        - Attention Time: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$
        - Space: $\mathcal{O}(B \cdot H \cdot N^2)$.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, N, D]`
        - Output: Tuple `(output, attn_weights)` where `output` is `[B, N, D]`.
    """

    def __init__(
        self,
        model_config: Any,
        energy: float = 0.95,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(model_config, rank)
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot initialisation")

        self.num_heads: int = getattr(model_config, "num_attention_heads", 12)
        hidden_size: int = getattr(model_config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size

        self.q_proj: FactorisedLinear = FactorisedLinear(
            pretrained_attn.q_lin.weight, pretrained_attn.q_lin.bias, energy
        )
        self.k_proj: FactorisedLinear = FactorisedLinear(
            pretrained_attn.k_lin.weight, pretrained_attn.k_lin.bias, energy
        )
        self.v_proj: FactorisedLinear = FactorisedLinear(
            pretrained_attn.v_lin.weight, pretrained_attn.v_lin.bias, energy
        )
        self.out_proj: FactorisedLinear = FactorisedLinear(
            pretrained_attn.out_lin.weight, pretrained_attn.out_lin.bias, energy
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass for MinimalSVDAttention.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional attention mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention matrix.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` where `output` is `[B, N, D]`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_heads
        head_dim = self.head_dim

        Q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, num_heads, head_dim)
            .transpose(1, 2)
        )

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_mask = attention_mask[:, None, None, :]
            else:
                extended_mask = attention_mask
            extended_mask = (1.0 - extended_mask.to(scores.dtype)) * -1e9
            scores = scores + extended_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality loss (0.0)."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_svd")
def inject_minimal_svd(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> MinimalSVDAttention:
    r"""
    Injection hook for MinimalSVDAttention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `MinimalSVDAttention` assigned to `layer.attention`.
    """
    energy: float = getattr(comp_config, "svd_energy", 0.95)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn = MinimalSVDAttention(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
