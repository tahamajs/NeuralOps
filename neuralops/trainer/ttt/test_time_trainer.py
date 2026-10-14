"""
Test-Time Training (TTT) Linear Attention & Meta-Learner.
Formulates the recurrent state as an online inner-loop model updated via
preconditioned gradient descent on reconstruction loss.
"""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("ttt_linear_attention")
class TTTLinearAttentionLayer(BaseAttention):
    """
    Test-Time Training (TTT) Attention Layer.
    Inner loop updates recurrent memory matrix S via online gradient descent:
        S_t = alpha * S_{t-1} - eta_t * (S_{t-1} @ k_t - v_t) @ k_t^T
        y_t = q_t @ S_t
    """

    def __init__(
        self,
        config: Any = None,
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        inner_lr: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        resolved_dim = dim or getattr(config, "hidden_size", 768)
        resolved_heads = num_heads or getattr(config, "num_attention_heads", 12)

        self.hidden_size = resolved_dim
        self.num_heads = resolved_heads
        self.head_dim = resolved_dim // resolved_heads
        self.inner_lr = inner_lr

        self.q_proj = nn.Linear(resolved_dim, resolved_dim, bias=False)
        self.k_proj = nn.Linear(resolved_dim, resolved_dim, bias=False)
        self.v_proj = nn.Linear(resolved_dim, resolved_dim, bias=False)
        self.out_proj = nn.Linear(resolved_dim, resolved_dim, bias=False)

        # Meta-learned token-dependent learning rate generator
        self.eta_net = nn.Sequential(
            nn.Linear(resolved_dim, resolved_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        etas = self.eta_net(hidden_states).view(B, N, H).transpose(1, 2).unsqueeze(-1) * self.inner_lr

        k_norm = F.normalize(k, p=2, dim=-1)

        # Online inner-loop state update (streaming across sequence)
        S = torch.zeros(B, H, d, d, device=hidden_states.device, dtype=torch.float32)
        outs = []

        for t in range(N):
            qt = q[:, :, t : t + 1].float()
            kt = k_norm[:, :, t : t + 1].float()
            vt = v[:, :, t : t + 1].float()
            eta = etas[:, :, t : t + 1].float()

            # Reconstruction error of online model: err = S_{t-1} @ k_t - v_t
            pred_v = torch.matmul(kt, S)
            err = pred_v - vt

            # Online gradient step on memory S
            grad_S = torch.matmul(kt.transpose(-1, -2), err)
            S = S - eta * grad_S

            # Readout
            yt = torch.matmul(qt, S)
            outs.append(yt.to(hidden_states.dtype))

        y = torch.cat(outs, dim=2).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(y), None


__all__ = ["TTTLinearAttentionLayer"]
