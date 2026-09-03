import math
from typing import Any, Dict, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class FactorisedLinear(nn.Module):
    r"""
    SVD-Factorized Linear Layer: $y = x W^\top + b$ where $W \approx A B$.

    Mathematical Mechanism:
        $$W = U \Sigma V^\top \approx U_{:, :r} \Sigma_{:r} V_{:r, :}^\top = A B$$
        $$A = U_{:, :r} \text{diag}(\sqrt{\Sigma_{:r}}) \in \mathbb{R}^{\text{out} \times r}, \quad B = \text{diag}(\sqrt{\Sigma_{:r}}) V_{:r, :}^\top \in \mathbb{R}^{r \times \text{in}}$$
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
        return self.up(self.down(x)), getattr(self, '_attn_weights', None)


@ATTENTION_REGISTRY.register("minimal_svd_v2")
class MinimalSVDAttentionV2(BaseAttention):
    r"""
    MinimalSVD v2: Clean SVD-Factorized Attention for Transformers.

    Mathematical Mechanism:
        Replaces projection matrices with truncated SVD low-rank factors:
        $$\begin{aligned}
        Q &= \text{FactorisedLinear}_Q(X) \\
        K &= \text{FactorisedLinear}_K(X) \\
        V &= \text{FactorisedLinear}_V(X) \\
        S &= \frac{Q K^\top}{\sqrt{d_k}} + M_{\text{mask}} \\
        A &= \text{softmax}(S, \text{dim}=-1) \\
        O &= \text{FactorisedLinear}_O(A V)
        \end{aligned}$$

    Computational Complexity:
        - Projections: $\mathcal{O}(B \cdot N \cdot r \cdot D)$
        - Attention: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$ time, $\mathcal{O}(B \cdot H \cdot N^2)$ space.

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
        self.num_heads: int = getattr(model_config, "num_attention_heads", 4)
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
        Forward pass for MinimalSVDAttentionV2.

        Args:
            hidden_states: Input tensor of shape `[B, N, D]`.
            attention_mask: Optional attention mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Boolean flag to return attention matrix.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, attn_weights)` where `output` is `[B, N, D]`.
        """
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -10000.0
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -10000.0
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None
            if token_valid is not None:
                scores = scores.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(2), -10000.0
                )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        output = self.out_proj(context)
        return output, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        r"""Returns orthogonality loss (0.0)."""
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_svd_v2_inject")
def inject_minimal_svd_v2(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
    **kwargs) -> MinimalSVDAttentionV2:
    r"""Injection hook for MinimalSVDAttentionV2 into transformer layer."""
    energy: float = getattr(comp_config, "svd_energy", 0.95)
    dropout: float = getattr(comp_config, "dropout", 0.1)
    new_attn = MinimalSVDAttentionV2(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
