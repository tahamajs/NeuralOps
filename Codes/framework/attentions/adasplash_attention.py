r"""
AdaSplash Attention Module with $\alpha$-Entmax Sparse Activations.

This module implements sparse attention via the 1.5-entmax mapping (Peters et al., 2019; Correia et al., 2019),
which yields exact zeros in attention weights, inducing sparse, interpretable attention distributions.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def entmax15(scores: torch.Tensor, dim: int = -1, n_iter: int = 10, eps: float = 1e-5) -> torch.Tensor:
    r"""
    Approximates the 1.5-entmax transformation via iterative fixed-point normalization.

    Mathematical Formulation:
        For score vector $z \in \mathbb{R}^N$, $\text{entmax}_{1.5}(z)$ is defined as:
        $$\text{entmax}_{1.5}(z) = \left[ \frac{1}{2} (z - \tau \mathbf{1}) \right]_+^2$$
        where $[\cdot]_+ = \max(\cdot, 0)$ and $\tau \in \mathbb{R}$ is the Lagrange multiplier ensuring $\sum_i p_i = 1$.
        This implementation utilizes an iterative fixed-point algorithm:
        $$q^{(t)} = \sqrt{p^{(t)}}, \quad \tau^{(t)} = \sum_i q_i^{(t)} p_i^{(t)}, \quad p^{(t+1)} = \frac{q^{(t)}}{\tau^{(t)}}$$

    Args:
        scores (torch.Tensor): Input attention logits of arbitrary shape with reduction along `dim`.
        dim (int): Dimension over which to compute entmax normalization (default: -1).
        n_iter (int): Maximum number of fixed-point iterations (default: 10).
        eps (float): Convergence threshold for absolute difference (default: 1e-5).

    Returns:
        torch.Tensor: Sparse probability distribution with identical shape to `scores`.
    """
    p = F.softmax(scores, dim=dim)
    for _ in range(n_iter):
        q = p.sqrt()
        tau = torch.sum(q * p, dim=dim, keepdim=True)
        p_next = q / (tau + 1e-5)
        if torch.max(torch.abs(p_next - p)) < eps:
            p = p_next
            break
        p = p_next
    return p


@ATTENTION_REGISTRY.register("adasplash")
class AdaSplashAttention(BaseAttention):
    r"""
    AdaSplash Attention layer leveraging Tsallis 1.5-entropy regularized sparse attention.

    Mechanism:
        1. Multi-head projection of queries, keys, and values:
           $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
        2. Scaled dot-product similarity computation:
           $$S = \frac{Q K^T}{\sqrt{d_k}} + M_{\text{mask}}$$
        3. Sparse activation mapping:
           $$A = \text{entmax}_{1.5}(S)$$
        4. Context aggregation and output projection:
           $$O = (A V) W_O$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h + B \cdot H \cdot L^2 \cdot T_{\text{iter}} + B \cdot L \cdot D^2)$
          where $B$ is batch size, $L$ is sequence length, $H$ is head count, $D$ is model dimension, $d_h = D/H$,
          and $T_{\text{iter}}$ is the iteration count of entmax.
        - Space Complexity: $\mathcal{O}(B \cdot H \cdot L^2)$ activation memory for attention matrices.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - `attention_mask`: $[B, L]$ or $[B, 1, 1, L]$
        - Projected $Q, K, V$: $[B, H, L, d_h]$
        - Attention Weights $A$: $[B, H, L, L]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        alpha: float = 1.5,
        n_iter: int = 10,
    ) -> None:
        super().__init__(config, rank)
        self.alpha: float = alpha
        self.n_iter: int = n_iter
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout if hasattr(config, "dropout") else 0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for AdaSplash Attention.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Binary or additive attention mask.
            output_attentions (bool): Whether to return sparse attention weights.
            **kwargs (Any): Additional configuration parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed hidden states of shape $[B, L, D]$.
                - Entmax attention weights $[B, H, L, L]$ if `output_attentions=True`, else None.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh**0.5)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            mask_float = mask.float()
            logits = logits + (1.0 - mask_float) * -10000.0

        if self.alpha == 1.5:
            attn = entmax15(logits, dim=-1, n_iter=self.n_iter)
        else:
            raise NotImplementedError(f"Alpha {self.alpha} is not currently supported; only 1.5 is implemented.")

        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), (attn if output_attentions else None)
