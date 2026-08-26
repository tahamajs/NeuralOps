r"""
TPAttention: Tensor Product Attention with Factorised Query-Key Projections.

Replaces standard linear projections for queries and keys with a Hadamard/outer
product of two low-rank factorised feature vectors, yielding richer polynomial
interaction terms at reduced per-token projection cost.

Mathematical Mechanism:
    Given input $X \in \mathbb{R}^{B \times L \times D}$ with $H$ heads and factorization rank $r$:
    1. **Factorised Projection:**
       $$q_1 = X W_{Q_1} \in \mathbb{R}^{B \times L \times H \times r}, \quad q_2 = X W_{Q_2} \in \mathbb{R}^{B \times L \times H \times r}$$
       $$k_1 = X W_{K_1} \in \mathbb{R}^{B \times L \times H \times r}, \quad k_2 = X W_{K_2} \in \mathbb{R}^{B \times L \times H \times r}$$
    2. **Tensor Product Expansion:**
       $$Q_{h, i} = q_{1, h, i} \otimes q_{2, h, i} \in \mathbb{R}^{r^2}, \quad K_{h, j} = k_{1, h, j} \otimes k_{2, h, j} \in \mathbb{R}^{r^2}$$
    3. **Scaled Dot-Product Attention:**
       $$A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{r}} + M\right) \in \mathbb{R}^{B \times H \times L \times L}$$
       $$O = (A V) W_O \in \mathbb{R}^{B \times L \times D}$$

Computational Complexity:
    - Factorised projection: $\mathcal{O}(B \cdot L \cdot D \cdot H \cdot r)$.
    - Tensor product expansion: $\mathcal{O}(B \cdot L \cdot H \cdot r^2)$.
    - Attention matrix products: $\mathcal{O}(B \cdot H \cdot L^2 \cdot r^2)$.
    - Total: $\mathcal{O}(B \cdot H \cdot L \cdot (D r + r^2 + L r^2))$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("tpa")
class TPAttention(BaseAttention):
    r"""Tensor Product Attention with factorised low-rank query-key projection.

    Query and key representations are formed as outer products of two
    factorised feature vectors, enabling richer polynomial similarity kernels.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and `dropout`.
        rank: Factorization rank $r$ per head (default 8). Effective Q/K dimension becomes $r^2$.
        pretrained_attn: Optional pretrained attention module for weight initialization.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 8,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.rank: int = rank
        self.dim: int = config.hidden_size
        dropout_p: float = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )

        # Factorised projections for Q and K (two parts each)
        self.q_proj1: nn.Linear = nn.Linear(self.dim, self.num_heads * self.rank)
        self.q_proj2: nn.Linear = nn.Linear(self.dim, self.num_heads * self.rank)
        self.k_proj1: nn.Linear = nn.Linear(self.dim, self.num_heads * self.rank)
        self.k_proj2: nn.Linear = nn.Linear(self.dim, self.num_heads * self.rank)
        # Value and output projections are standard
        self.v_proj: nn.Linear = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.out_proj: nn.Linear = nn.Linear(self.num_heads * self.head_dim, self.dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Initialise the standard projections from a pretrained attention module."""
        with torch.no_grad():
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)
            # Factorised Q/K: small random initialisation
            for proj1, proj2 in [
                (self.q_proj1, self.q_proj2),
                (self.k_proj1, self.k_proj2),
            ]:
                nn.init.normal_(proj1.weight, std=0.02)
                nn.init.normal_(proj2.weight, std=0.02)

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
        r"""Compute tensor product attention.

        Args:
            hidden_states: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask `[Batch, SeqLen]` or broadcastable additive mask.
            head_mask: Optional per-head gating mask.
            output_attentions: Return explicit attention weights if True.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected representation `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Softmax probability matrix or None.
        """
        B, L, D = hidden_states.shape
        H, r, hd = self.num_heads, self.rank, self.head_dim

        q1 = self.q_proj1(hidden_states).view(B, L, H, r)
        q2 = self.q_proj2(hidden_states).view(B, L, H, r)
        k1 = self.k_proj1(hidden_states).view(B, L, H, r)
        k2 = self.k_proj2(hidden_states).view(B, L, H, r)
        v = self.v_proj(hidden_states).view(B, L, H, hd)

        # Tensor product: Q = q1 ⊗ q2, shape [B, L, H, r*r]
        Q = torch.einsum("blhi,blhj->blhij", q1, q2).reshape(B, L, H, r * r)
        K = torch.einsum("blhi,blhj->blhij", k1, k2).reshape(B, L, H, r * r)

        Q = Q.permute(0, 2, 1, 3)  # (B, H, L, r*r)
        K = K.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # (B, H, L, hd)

        scale = r**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, L, L)

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

        out = torch.matmul(attn, v)  # (B, H, L, hd)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn
        return out, None


def inject_tpa(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> TPAttention:
    r"""Bridge helper injecting TPAttention into a transformer layer.

    Returns:
        The newly instantiated :class:`TPAttention`.
    """
    rank = getattr(comp_config, "rank", 8)
    new_attn = TPAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
