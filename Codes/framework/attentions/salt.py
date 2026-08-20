r"""
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.

Parameter-efficient adaptation mechanism for transformer attention projections combining
orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$).

Mathematical Mechanism:
    For a pretrained projection weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with SVD $W_0 = U \Sigma V^T$:
    1. **Singular Value Rescaling & Adapter Fusion:**
       $$W = U_{:, :r} \operatorname{diag}(\Sigma_{1:r} \odot \gamma) V_{1:r, :}^T + A B$$
       where $U_{:, :r}, V_{1:r, :}^T$ are frozen orthogonal bases, $\gamma \in \mathbb{R}^r$ is a learnable scaling vector,
       and $A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{r \times d_{\text{in}}}$ form a low-rank residual branch.
    2. **Multi-Head Attention Computation:**
       $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
       $$\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$

Computational Complexity:
    - Weight reconstruction / forward projection: $\mathcal{O}(B \cdot L \cdot (d_{\text{in}} r + r d_{\text{out}}))$.
    - Attention matrix evaluation: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d)$.
    - Trainable parameter footprint per projection is reduced to $\mathcal{O}(r \cdot (d_{\text{in}} + d_{\text{out}}) + r)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SALTLinear(nn.Module):
    r"""Linear projection layer with SVD singular value modulation and low-rank residual branch.

    Args:
        original_weight: Source weight tensor of shape `[OutFeatures, InFeatures]`.
        rank: Adaptation rank $r$.
    """

    def __init__(self, original_weight: torch.Tensor, rank: int = 16) -> None:
        super().__init__()
        out_features, in_features = original_weight.shape
        U, S, Vt = torch.linalg.svd(original_weight.float(), full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("Vt", Vt)
        self.S = nn.Parameter(S)
        self.gamma = nn.Parameter(torch.ones(rank))
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute adapted linear projection: $y = x (U \operatorname{diag}(S \odot \gamma) V^T + A B)^T$."""
        r = self.gamma.shape[0]
        adapted_S = self.S[:r] * self.gamma
        adapted_weight = (
            self.U[:, :r]
            @ torch.diag(adapted_S)
            @ self.Vt[:r, :]
        )
        low_rank = self.A @ self.B
        W = adapted_weight + low_rank
        return F.linear(x, W)


@ATTENTION_REGISTRY.register("salt_attention")
class SALTAttention(BaseAttention):
    r"""SALT-adapted attention module replacing standard linear projections with SALTLinear.

    Args:
        config: Transformer configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Adaptation rank $r$.
        pretrained_attn: Pretrained attention sub-layer from which weights are initialized.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 16,
        pretrained_attn: Optional[Any] = None,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        dropout_p: float = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
        )

        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for SALTAttention initialization.")

        self.q_proj = SALTLinear(pretrained_attn.q_lin.weight.data, rank)
        self.k_proj = SALTLinear(pretrained_attn.k_lin.weight.data, rank)
        self.v_proj = SALTLinear(pretrained_attn.v_lin.weight.data, rank)
        self.out_proj = SALTLinear(pretrained_attn.out_lin.weight.data, rank)

        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward execution of SALT attention.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor `[Batch, SeqLen]` or broadcastable.
            head_mask: Optional head gating mask.
            output_attentions: Return explicit attention probabilities if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention matrix or None.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        scale = Dh**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores + (1.0 - mask) * -10000.0
            else:
                scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_salt_attention(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SALTAttention:
    r"""Bridge helper injecting SALTAttention into a target transformer layer."""
    rank = getattr(comp_config, "rank", 16)
    new_attn = SALTAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
