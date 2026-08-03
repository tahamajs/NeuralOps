"""
Performer Attention – Fast Attention Via Positive Orthogonal Random Features (FAVOR+).
Paper: https://arxiv.org/abs/2009.14794
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(nn.Module):
    r"""Performer: Fast Attention via Positive Orthogonal Random Features (FAVOR+) (Choromanski et al., 2021).

    Linearizes the softmax kernel using unbiased positive orthogonal random feature maps:
    $$\phi(x) = \frac{1}{\sqrt{m}} \exp\left(\frac{x \Omega^\top}{\sqrt{d_k}} - \frac{\|x\|^2}{2 \sqrt{d_k}}\right)$$
    $$\text{Attention}_{\text{FAVOR+}}(Q, K, V) = D^{-1} \left( \phi(Q) \left(\phi(K)^\top V\right) \right), \quad D = \operatorname{diag}\left(\phi(Q) \left(\phi(K)^\top \mathbf{1}_N\right)\right)$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot m \cdot d_k)$ (strictly linear in sequence length $N$)
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot m + m \cdot d_k))$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.num_features: int = getattr(
            config,
            "performer_rank",
            rank if rank is not None else getattr(config, "rank", 64),
        )
        self.eps: float = eps

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)

        # Aliases
        # Register orthogonal random projection matrix
        omega = self._get_orthogonal_projection_matrix(self.head_dim, self.num_features)
        self.register_buffer("omega", omega)

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

    def _get_orthogonal_projection_matrix(self, d: int, m: int) -> torch.Tensor:
        """Construct orthogonal random features by stacking QR decomposed Gaussian blocks."""
        num_blocks = math.ceil(m / d)
        blocks = []
        for _ in range(num_blocks):
            g = torch.randn(d, d)
            q, _ = torch.linalg.qr(g)
            blocks.append(q.T)
        matrix = torch.cat(blocks, dim=0)[:m]  # (m, d)
        # Multiplier according to FAVOR+ theory
        multiplier = torch.randn(m, d).norm(dim=1, keepdim=True)
        return matrix * multiplier

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute positive orthogonal random feature map $\phi(x)$ with numerical stabilization."""
        # x: (B, H, N, d)
        # omega: (m, d) -> x @ omega.T: (B, H, N, m)
        proj = torch.matmul(x, self.omega.to(dtype=x.dtype, device=x.device).T)
        x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True) / 2.0  # (B, H, N, 1)

        # Numerical stabilization
        logits = proj - x_norm_sq
        max_val = logits.amax(dim=-1, keepdim=True).detach()
        phi = torch.exp(logits - max_val) / math.sqrt(self.num_features)
        return phi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for FAVOR+ linear attention."""
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Pre-scale queries and keys by d^-1/4
        scale = d**-0.25
        Q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2) * scale
        K: torch.Tensor = self.key(hidden_states).view(B, N, H, d).transpose(1, 2) * scale
        V: torch.Tensor = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)  # (B, 1, N, 1)

        # Feature maps: (B, H, N, m)
        Q_prime = self._feature_map(Q)
        K_prime = self._feature_map(K) * mask_expanded

        # Associative linear multiplication: O(N * m * d)
        # KV: (B, H, m, d)
        KV = torch.matmul(K_prime.transpose(-2, -1), V)

        # Normalization denominator: (B, H, N, 1)
        K_sum = K_prime.sum(dim=-2, keepdim=True)  # (B, H, 1, m)
        den = torch.matmul(Q_prime, K_sum.transpose(-2, -1)) + self.eps  # (B, H, N, 1)

        # Context: (B, H, N, d)
        num = torch.matmul(Q_prime, KV)
        context = num / den

        context = self.dropout(context)
        if head_mask is not None:
            context = context * head_mask

        # Mask padding queries
        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out: torch.Tensor = self.out_proj(context)

        return out, None
