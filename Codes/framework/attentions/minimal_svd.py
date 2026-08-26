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
        $$A = U_{:, :r} \text{diag}(\sqrt{S_{:r}}) \in \mathbb{R}^{\text{out} \times r}, \quad B = \text{diag}(\sqrt{S_{:r}}) V_{:r, :} \in \mathbb{R}^{r \times \text{in}}$$
        $$y = (x B^\top) A^\top + b$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        energy: float = 0.95,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy

        self.down: nn.Linear = nn.Linear(in_features, self.rank, bias=False)
        self.up: nn.Linear = nn.Linear(self.rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize with orthogonal factors."""
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        """Decompose pretrained dense weight matrix via SVD and set factor parameters."""
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_energy = energy if energy is not None else self.energy
            total_energy = (S**2).sum()
            cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)
            calc_rank = int(torch.searchsorted(cum_energy, target_energy).item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            sqrt_s = torch.sqrt(S[:r])
            A = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
            B = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(self.down.weight.dtype)

            if r != self.rank:
                # Pad to buffer shape if rank differs
                self.up.weight.data.zero_()
                self.down.weight.data.zero_()
                self.up.weight.data[:, :r] = A
                self.down.weight.data[:r, :] = B
            else:
                self.up.weight.data.copy_(A)
                self.down.weight.data.copy_(B)

            if bias is not None and self.up.bias is not None:
                self.up.bias.data.copy_(bias)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Evaluates factorized linear transformation."""
        return self.up(self.down(x))


@ATTENTION_REGISTRY.register("minimal_svd")
class MinimalSVDAttention(BaseAttention):
    r"""
    MinimalSVD: SVD-Factorized Low-Rank Projection Attention for Transformers.

    Mathematical Mechanism:
        Compresses attention projections using truncated singular value factorizations
        while maintaining full quadratic attention:
        $$\begin{aligned}
        Q &= \text{FactorisedLinear}_Q(X) \\
        K &= \text{FactorisedLinear}_K(X) \\
        V &= \text{FactorisedLinear}_V(X) \\
        S &= \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{mask}} \\
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
        **kwargs: Any,
    ) -> None:
        actual_rank = getattr(
            model_config,
            "svd_rank",
            rank if rank is not None else getattr(model_config, "rank", 64),
        )
        super().__init__(model_config, actual_rank)

        self.num_heads: int = getattr(model_config, "num_attention_heads", 12)
        hidden_size: int = getattr(model_config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.hidden_size: int = hidden_size
        self.rank: int = actual_rank
        self.energy: float = energy

        self.q_proj: FactorisedLinear = FactorisedLinear(
            self.hidden_size, self.all_head_size, rank=self.rank, bias=True, energy=energy
        )
        self.k_proj: FactorisedLinear = FactorisedLinear(
            self.hidden_size, self.all_head_size, rank=self.rank, bias=True, energy=energy
        )
        self.v_proj: FactorisedLinear = FactorisedLinear(
            self.hidden_size, self.all_head_size, rank=self.rank, bias=True, energy=energy
        )
        self.out_proj: FactorisedLinear = FactorisedLinear(
            self.all_head_size, self.hidden_size, rank=self.rank, bias=True, energy=energy
        )

        dropout_p = getattr(
            model_config,
            "dropout",
            getattr(
                model_config,
                "hidden_dropout_prob",
                getattr(model_config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [
            ("q", self.q_proj),
            ("k", self.k_proj),
            ("v", self.v_proj),
            ("o", self.out_proj),
        ]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.decompose_from_weight(w, b, self.energy)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

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
            elif attention_mask.dim() == 3:
                extended_mask = attention_mask[:, None, :, :]
            else:
                extended_mask = attention_mask
            extended_mask = (1.0 - extended_mask.to(scores.dtype)) * -10000.0
            scores = scores + extended_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality loss (0.0)."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_svd_inject")
def inject_minimal_svd(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
    **kwargs) -> MinimalSVDAttention:
    r"""Injection hook for MinimalSVDAttention into transformer layer."""
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
