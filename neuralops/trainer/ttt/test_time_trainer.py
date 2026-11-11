#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test-Time Training Engine

Implements test-time training (TTT) and dynamic sequence-level model adaptation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


def _ttt_linear_online_gd_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    etas: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Chunked vectorized PyTorch implementation of exact TTT online gradient descent."""
    B, H, N, d = q.shape
    if N == 0:
        return torch.empty_like(q)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    etas_f = etas.float()
    if etas_f.dim() == 3:
        etas_f = etas_f.unsqueeze(-1)

    S = torch.zeros(B, H, d, d, device=q.device, dtype=torch.float32)
    outs = []
    use_triangular = hasattr(torch.linalg, "solve_triangular")

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        c_len = end - start
        q_c = q_f[:, :, start:end, :]
        k_c = k_f[:, :, start:end, :]
        v_c = v_f[:, :, start:end, :]
        eta_c = etas_f[:, :, start:end, :]

        if use_triangular:
            pred_v = torch.matmul(k_c, S)
            rhs = eta_c * (v_c - pred_v)
            kk = torch.matmul(k_c, k_c.transpose(-1, -2))
            lower_strict = torch.tril(eta_c * kk, diagonal=-1)
            eye = torch.eye(c_len, device=q.device, dtype=torch.float32)
            updates = torch.linalg.solve_triangular(eye + lower_strict, rhs, upper=False, unitriangular=True)

            qk = torch.tril(torch.matmul(q_c, k_c.transpose(-1, -2)), diagonal=0)
            y_c = torch.matmul(q_c, S) + torch.matmul(qk, updates)
            S = S + torch.matmul(k_c.transpose(-1, -2), updates)
            outs.append(y_c.to(q.dtype))
        else:
            for t in range(c_len):
                qt = q_c[:, :, t : t + 1]
                kt = k_c[:, :, t : t + 1]
                vt = v_c[:, :, t : t + 1]
                eta = eta_c[:, :, t : t + 1]
                err = torch.matmul(kt, S) - vt
                S = S - eta * torch.matmul(kt.transpose(-1, -2), err)
                outs.append(torch.matmul(qt, S).to(q.dtype))

    return torch.cat(outs, dim=2)


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
        resolved_dim = (
            dim
            or kwargs.get("hidden_size")
            or kwargs.get("embed_dim")
            or kwargs.get("d_model")
            or getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
            or 768
        )
        resolved_heads = (
            num_heads
            or kwargs.get("num_attention_heads")
            or kwargs.get("n_heads")
            or getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
            or 12
        )

        self.hidden_size = int(resolved_dim)
        self.num_heads = int(resolved_heads)
        self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.all_head_size = self.num_heads * self.head_dim
        self.inner_lr = inner_lr

        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=False)

        self.eta_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_heads),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        etas = self.eta_net(hidden_states).view(B, N, H).transpose(1, 2).unsqueeze(-1) * self.inner_lr

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                etas = etas * (attention_mask > 0).to(etas.dtype).view(B, 1, N, 1)
            elif attention_mask.dim() == 4 and attention_mask.shape[2] == 1:
                etas = etas * (attention_mask[:, 0, 0, :] > -1.0).to(etas.dtype).view(B, 1, N, 1)

        k_norm = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

        # Use vectorized chunked PyTorch TTT online GD directly to avoid blocking
        # on slow nvcc/Triton JIT compilation during discovery and short runs.
        y_heads = _ttt_linear_online_gd_pytorch(q, k_norm, v, etas)

        y = y_heads.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        return self.out_proj(y), None


__all__ = ["TTTLinearAttentionLayer"]
