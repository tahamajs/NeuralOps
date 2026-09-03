r"""
Zero-Mean Linear Attention (ZeroS Attention).

Removes the mean from key representations and applies normalized
linear attention with ReLU-based non-negative feature maps.

Mathematical Mechanism:
    For input $X \in \mathbb{R}^{B \times L \times D}$ with $H$ heads:
    1. **Zero-Centering of Keys:**
       $$\tilde{K} = K - \frac{1}{L} \sum_{i=1}^{L} K_i \in \mathbb{R}^{B \times H \times L \times d}$$
    2. **Non-Negative Feature Map:**
       $$\hat{Q} = \text{ReLU}(Q) + \epsilon, \quad \hat{K} = \text{ReLU}(\tilde{K}) + \epsilon$$
    3. **Normalized Linear Attention:**
       $$S_{KV} = \hat{K}^\top V \in \mathbb{R}^{B \times H \times d \times d}$$
       $$\text{Num} = \hat{Q} S_{KV} \in \mathbb{R}^{B \times H \times L \times d}$$
       $$\text{Den} = \hat{Q} \left(\sum_{i=1}^{L} \hat{K}_i\right)^\top + \epsilon$$
       $$O = \frac{\text{Num}}{\text{Den}} \in \mathbb{R}^{B \times H \times L \times d}$$

Computational Complexity:
    - Time: $\mathcal{O}(B \cdot H \cdot L \cdot d^2)$ — strictly linear $\mathcal{O}(L)$ in sequence length.
    - Space: $\mathcal{O}(B \cdot H \cdot d^2)$ for KV accumulator.
"""

from typing import Any, Optional, Tuple

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("zeros")
class ZeroSAttention(BaseAttention):
    r"""Zero-mean normalized linear attention module.

    Applies sequence-mean centering on key projections before
    running efficient non-negative linear attention.

    Args:
        config: Model configuration specifying `hidden_size` and `num_attention_heads`.
        rank: Optional rank parameter (for BaseAttention compatibility).
    """

    def __init__(self, config: Any, rank: int = 64) -> None:
        super().__init__(config, rank)
        self.head_dim: int = config.hidden_size // config.num_attention_heads
        self.num_heads: int = config.num_attention_heads
        self.W_q: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_k: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_v: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.out: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)

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
        r"""Compute zero-mean linear attention.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional mask tensor `[Batch, SeqLen]` or broadcastable.
            head_mask: Unused; for interface compatibility.
            output_attentions: Unused; for interface compatibility.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of:
                - output: Attended representation `[Batch, SeqLen, HiddenDim]`.
                - None: No explicit attention weights returned.
        """
        B, L, D = hidden_states.shape
        q = (
            self.W_q(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.W_k(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.W_v(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Zero-center keys along sequence dimension
        k_centered = k - k.mean(dim=2, keepdim=True)

        # Linear attention: O = Q (K^T V) / (Q sum(K))
        q_nonneg = torch.relu(q) + 1e-5
        k_nonneg = torch.relu(k_centered) + 1e-5

        kv = torch.einsum("bhnk,bhnd->bhkd", k_nonneg, v)  # (B, H, d, d)
        k_sum = k_nonneg.sum(dim=2, keepdim=True)  # (B, H, 1, d)

        num = torch.matmul(q_nonneg, kv)  # (B, H, L, d)
        den = torch.matmul(q_nonneg, k_sum.transpose(-2, -1)) + 1e-5
        out = num / den
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out(out), None
