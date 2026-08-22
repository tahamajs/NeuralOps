r"""
CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

Implements linear attention via spherical normalization onto the unit hypersphere
combined with trigonometric random Fourier feature mappings (Yat-Kernel).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("chela_slay")
class ChelaSlayAttention(BaseAttention):
    r"""
    CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

    Mathematical Formulation:
        1. Spherical Normalization:
           Projects queries and keys to the unit hypersphere $\mathbb{S}^{d_h - 1}$:
           $$\hat{Q} = \frac{Q}{\|Q\|_2}, \quad \hat{K} = \frac{K}{\|K\|_2}$$

        2. Yat-Kernel Feature Map:
           Computes trigonometric Fourier projections using random matrix $\Omega \in \mathbb{R}^{d_h \times F}$:
           $$\phi(x) = \frac{1}{F^{1/4}} \left[ \cos(x \Omega) \,\|\, \sin(x \Omega) \right] \in \mathbb{R}^{2F}$$

        3. Linear Attention Aggregation:
           Evaluates exact matrix associative reduction:
           $$KV = \sum_{j=1}^L \phi(\hat{K}_j)^T V_j \in \mathbb{R}^{2F \times d_h}$$
           $$Z = \sum_{j=1}^L \phi(\hat{K}_j)^T \in \mathbb{R}^{2F \times 1}$$
           $$O_i = \frac{\phi(\hat{Q}_i) KV}{\phi(\hat{Q}_i) Z + \epsilon}$$

    Complexity:
        - Time Complexity: $\mathcal{O}(B \cdot H \cdot L \cdot F \cdot d_h + B \cdot L \cdot D^2)$ linear in sequence length $L$.
        - Space Complexity: $\mathcal{O}(B \cdot H \cdot F \cdot d_h)$ recurrent memory buffer.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Spherical Feature Map: $[B, H, L, 2F]$
        - State Matrix $KV$: $[B, H, 2F, d_h]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        feature_dim: int = 256,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.dim // self.num_heads
        self.feature_dim: int = feature_dim

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.register_buffer("random_proj", torch.randn(self.head_dim, feature_dim))

    def _spherical_normalize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Applies Euclidean $\ell_2$ normalization across the head dimension."""
        return F.normalize(x, p=2, dim=-1)

    def _yat_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluates the Yat-kernel random feature map $\phi(x)$.

        Args:
            x (torch.Tensor): Input spherical tensor of shape $[*, d_h]$.

        Returns:
            torch.Tensor: Feature map tensor of shape $[*, 2F]$.
        """
        proj = torch.matmul(x, self.random_proj)  # (..., feature_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (
            1.0 / (self.feature_dim**0.25)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for ChelaSlayAttention.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, L]$.
            output_attentions (bool): Flag indicating whether to return attention maps.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context representations $[B, L, D]$.
                - None.
        """
        x = hidden_states
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q_norm = self._spherical_normalize(q)
        k_norm = self._spherical_normalize(k)

        q_feat = self._yat_feature_map(q_norm)  # (B, H, L, 2F)
        k_feat = self._yat_feature_map(k_norm)  # (B, H, L, 2F)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, L) -> (B, 1, L, 1)
                mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(k_feat.dtype)
            elif attention_mask.dim() == 4:
                # HF passes (B, 1, 1, L) — reshape to (B, 1, L, 1)
                mask = (attention_mask[:, 0, 0, :] > -10000.0).float().unsqueeze(1).unsqueeze(-1).to(k_feat.dtype)
            else:
                mask = attention_mask.to(k_feat.dtype).unsqueeze(-1)
            k_feat = k_feat * mask
            v = v * mask

        kv = torch.matmul(k_feat.transpose(-2, -1), v)  # (B, H, 2F, D_h)
        k_sum = k_feat.sum(dim=2, keepdim=True)  # (B, H, 1, 2F)
        denom = torch.matmul(q_feat, k_sum.transpose(-2, -1)) + 1e-5  # (B, H, L, 1)
        out = torch.matmul(q_feat, kv) / denom
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None
