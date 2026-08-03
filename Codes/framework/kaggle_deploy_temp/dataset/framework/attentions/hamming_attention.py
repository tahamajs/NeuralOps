import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hamming")
class HammingAttention(BaseAttention):
    r"""
    Hamming Attention Distillation (HAD) Module.

    Mathematical Mechanism:
        Approximates dense inner products with differentiable continuous relaxations of Hamming distances:
        $$\begin{aligned}
        Q_b &= \tanh(5.0 \cdot \gamma_Q \odot Q) \in [-1, 1]^{B \times H \times L \times d_k} \\
        K_b &= \tanh(5.0 \cdot \gamma_K \odot K) \in [-1, 1]^{B \times H \times L \times d_k} \\
        D_H(Q_b, K_b) &= \frac{d_k - Q_b K_b^\top}{2} \\
        S &= -\frac{D_H(Q_b, K_b)}{\sqrt{d_k}} = \frac{Q_b K_b^\top - d_k}{2\sqrt{d_k}} \\
        P &= \text{softmax}(\text{TopK-Sparsify}(S, \text{ratio})) \\
        O &= P V W_O
        \end{aligned}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_k)$
        - Space: $\mathcal{O}(B \cdot H \cdot L^2)$ for score matrices.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        distill_temperature: float = 1.0,
        sparsity_ratio: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.distill_temperature: float = distill_temperature
        self.sparsity_ratio: float = sparsity_ratio

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        dropout_p: float = getattr(config, "dropout", 0.0)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        self.q_scale: nn.Parameter = nn.Parameter(torch.ones(self.num_heads, 1))
        self.k_scale: nn.Parameter = nn.Parameter(torch.ones(self.num_heads, 1))

    @staticmethod
    def _binarize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        r"""
        Differentiable tanh approximation of the sign/binarization operator.

        Args:
            tensor: Input tensor of shape `[B, H, L, d]`.
            scale: Head-wise scale factor `[H, 1]`.

        Returns:
            Continuous pseudo-binary tensor of shape `[B, H, L, d]`.
        """
        t = tensor * scale.view(1, -1, 1, 1)
        return torch.tanh(t * 5.0)

    @staticmethod
    def _hamming_dist(q_b: torch.Tensor, k_b: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates pairwise relaxed Hamming distance between pseudo-binary vectors.

        Args:
            q_b: Binarized query tensor `[B, H, L, d]`.
            k_b: Binarized key tensor `[B, H, L, d]`.

        Returns:
            Distance matrix of shape `[B, H, L, L]`.
        """
        dot = torch.matmul(q_b, k_b.transpose(-2, -1))
        return (q_b.shape[-1] - dot) / 2.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        teacher_attention: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass of HammingAttention.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional attention mask tensor `[B, L]` or `[B, 1, L, L]`.
            head_mask: Optional head mask tensor.
            output_attentions: Boolean flag indicating if attention map is returned.
            teacher_attention: Optional teacher attention matrix for distillation.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, probs)` where `output` is `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        q_b = self._binarize(q, self.q_scale)
        k_b = self._binarize(k, self.k_scale)

        scores = -self._hamming_dist(q_b, k_b) / (math.sqrt(Dh))

        # Mask handling (convert 2D mask to additive)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        if self.sparsity_ratio > 0:
            keep_k = max(1, int(L * self.sparsity_ratio))
            _, top_idx = torch.topk(scores, k=keep_k, dim=-1)
            sparse_mask = torch.zeros_like(scores).scatter(-1, top_idx, 1.0)
            scores = scores * sparse_mask + (1.0 - sparse_mask) * -1e9

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        out = self.out_proj(context)
        return out, (probs if output_attentions else None)
