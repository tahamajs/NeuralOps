"""
Attention Stability Regularizers:
1. LearnableAttentionSink (absorbs no-op attention mass in transplanted models)
2. AttentionEntropyRegularizer (penalizes entropy collapse)
3. GateSaturationPenalty (prevents early decay gate saturation)
4. SpectralJacobianRegularizer (bounds Lipschitz constants)
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableAttentionSink(nn.Module):
    """
    Learnable Sink Key-Value Tokens for linear/kernelized attention.
    Prepends sink tokens to prevent massive activation drift and absorb residual attention mass.
    """

    def __init__(self, num_heads: int, head_dim: int, num_sink_tokens: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_sinks = num_sink_tokens

        self.sink_k = nn.Parameter(torch.randn(1, num_heads, num_sink_tokens, head_dim) * 0.02)
        self.sink_v = nn.Parameter(torch.zeros(1, num_heads, num_sink_tokens, head_dim))

    def prepend_sinks(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = k.size(0)
        sk = self.sink_k.expand(B, -1, -1, -1)
        sv = self.sink_v.expand(B, -1, -1, -1)
        return torch.cat([sk, k], dim=-2), torch.cat([sv, v], dim=-2)


class GateSaturationPenalty(nn.Module):
    """Penalizes forget/decay gates g in (0, 1) saturating to 0 or 1 early in training."""

    def __init__(self, penalty_weight: float = 0.01) -> None:
        super().__init__()
        self.weight = penalty_weight

    def forward(self, gate_activations: torch.Tensor) -> torch.Tensor:
        # Penalize -log(g * (1 - g))
        g = gate_activations.clamp(min=1e-5, max=1.0 - 1e-5)
        barrier = -torch.log(g * (1.0 - g)).mean()
        return self.weight * barrier


class SpectralJacobianRegularizer(nn.Module):
    """2-Step GPU Power Iteration to bound max singular value of projection matrices."""

    def __init__(self, max_sigma: float = 5.0, weight: float = 0.05) -> None:
        super().__init__()
        self.max_sigma = max_sigma
        self.weight = weight

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        w = weight_matrix.float()
        # Random vector for power iteration
        v = torch.randn(w.size(1), 1, device=w.device, dtype=torch.float32)
        v = v / v.norm(p=2).clamp(min=1e-8)

        # 2 steps power iteration
        u = torch.matmul(w, v)
        u = u / u.norm(p=2).clamp(min=1e-8)
        v = torch.matmul(w.t(), u)
        sigma = v.norm(p=2)

        violation = F.relu(sigma - self.max_sigma)
        return self.weight * (violation ** 2)


__all__ = [
    "LearnableAttentionSink",
    "GateSaturationPenalty",
    "SpectralJacobianRegularizer",
]
