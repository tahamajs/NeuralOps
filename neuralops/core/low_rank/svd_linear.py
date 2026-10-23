#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical SVD Linear Implementation.

Provides the universal SVD-factorized linear layer with optional singular value
scaling, LoRA adapters, RMSNorm, and zero-overhead CPU/GPU fallback dispatch.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from neuralops.core.triton_kernels._base import HAS_TRITON, triton_dispatch
except ImportError:
    HAS_TRITON = False
    triton_dispatch = None


class SVDLinear(nn.Module):
    r"""
    Universal SVD-factorized linear layer:
        y = (x @ V.T) @ (U.T * S) + bias + (lora_alpha / lora_rank) * (x @ A.T) @ B.T
    """

    is_custom_attention: bool = False
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | None = None,
        bias: bool = True,
        freeze: bool = False,
        freeze_base: bool = False,
        freeze_svd: bool = False,
        lora_rank: int | None = 16,
        lora_alpha: float = 32.0,
        lora_r: int | None = None,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        use_triton: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank) if rank is not None else max(1, min(self.in_features, self.out_features) // 4)
        self.freeze_svd = bool(freeze or freeze_base or freeze_svd)
        eff_lora = int(lora_r) if (lora_r is not None and lora_r > 0) else (int(lora_rank) if lora_rank is not None else 0)
        self.lora_rank = max(0, eff_lora)
        self.lora_r = self.lora_rank
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / max(1, self.lora_rank)) if self.lora_rank > 0 else 1.0
        self.scaling = self.lora_scaling
        self.use_fused_kernel = bool(use_fused_kernel)
        self.use_rms_norm = bool(use_rms_norm)
        self.use_triton = bool(use_triton)
        self.eps = float(eps)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.U = nn.Parameter(torch.empty(self.out_features, self.rank, **factory_kwargs))
        self.S = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
        self.V = nn.Parameter(torch.empty(self.rank, self.in_features, **factory_kwargs))

        if self.use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
        else:
            self.register_parameter("norm_weight", None)

        if self.freeze_svd:
            self.U.requires_grad = False
            self.S.requires_grad = False
            self.V.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.lora_rank, **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory_kwargs))
            if self.freeze_svd:
                self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.U, self.V):
            orig_dtype = param.data.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.float()
                nn.init.orthogonal_(param)
                param.data = param.data.to(orig_dtype)
            else:
                nn.init.orthogonal_(param)
        if self.S is not None:
            nn.init.ones_(self.S)
        if self.norm_weight is not None:
            nn.init.ones_(self.norm_weight)
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, len(S))
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[: min(self.out_features, U.size(0)), :r].copy_(
                U[: self.out_features, :r].to(self.U.dtype)
            )
            self.S.data[:r].copy_(S[:r].to(self.S.dtype))
            self.V.data[:r, : min(self.in_features, Vt.size(1))].copy_(
                Vt[:r, : self.in_features].to(self.V.dtype)
            )

            if bias is not None and self.bias is not None:
                min_b = min(self.out_features, bias.size(0))
                self.bias.data.zero_()
                self.bias.data[:min_b].copy_(bias[:min_b])

    @property
    def weight(self) -> torch.Tensor:
        u = self.U
        if self.S is not None:
            u = u * self.S.unsqueeze(0)
        w = torch.matmul(u, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    def get_weight(self) -> torch.Tensor:
        return self.weight

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        with torch.no_grad():
            r = min(self.rank, U_factor.size(1), V_factor.size(0))
            if r < self.rank:
                logger.debug(
                    "SVDLinear: factor rank %d < allocated rank %d; zeroing remainder.",
                    r,
                    self.rank,
                )
            out_d = min(self.out_features, U_factor.size(0))
            in_d = min(self.in_features, V_factor.size(1))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:out_d, :r].copy_(U_factor[:out_d, :r].to(self.U.device, self.U.dtype))
            self.V.data[:r, :in_d].copy_(V_factor[:r, :in_d].to(self.V.device, self.V.dtype))

    def get_orthogonality_loss(self) -> torch.Tensor:
        U_orth = torch.matmul(self.U.t().contiguous(), self.U) - torch.eye(
            self.rank, device=self.U.device, dtype=self.U.dtype
        )
        V_orth = torch.matmul(self.V, self.V.t().contiguous()) - torch.eye(
            self.rank, device=self.V.device, dtype=self.V.dtype
        )
        return torch.norm(U_orth, p="fro") + torch.norm(V_orth, p="fro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        Vx = F.linear(x, self.V.to(target_dtype))
        if self.S is not None:
            Vx = Vx * self.S.to(target_dtype)
        if self.use_rms_norm and self.norm_weight is not None:
            norm = torch.rsqrt(Vx.pow(2).mean(-1, keepdim=True) + self.eps)
            Vx = Vx * norm * self.norm_weight.to(target_dtype)

        b = self.bias.to(target_dtype) if self.bias is not None else None
        out = F.linear(Vx, self.U.to(target_dtype), b)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x, self.lora_A.to(target_dtype)),
                self.lora_B.to(target_dtype),
            ) * self.lora_scaling
            out = out + lora_out

        return out.to(target_dtype)


__all__ = ["SVDLinear"]
