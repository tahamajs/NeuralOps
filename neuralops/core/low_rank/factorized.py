#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical Factorized Linear Implementation.

Provides the universal low-rank decomposed linear layer (W ~ W2 @ W1) with
optional LoRA residual adapters, energy-based truncation, and both American
(FactorizedLinear) and British (FactorisedLinear) naming aliases.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FactorizedLinear(nn.Module):
    """
    Decomposed linear layer: W ~ W2 @ W1 (or up @ down) with optional LoRA residual branch.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features_or_weight: Any = None,
        out_features_or_bias: Any = None,
        in_features: int | None = None,
        out_features: int | None = None,
        rank: int | None = None,
        svd_rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Resolve in_features and out_features
        if in_features is None and in_features_or_weight is not None:
            if isinstance(in_features_or_weight, (int, float)):
                in_features = int(in_features_or_weight)
            elif isinstance(in_features_or_weight, torch.Tensor):
                in_features = in_features_or_weight.shape[1]
                out_features = in_features_or_weight.shape[0]

        if out_features is None and out_features_or_bias is not None:
            if isinstance(out_features_or_bias, (int, float)):
                out_features = int(out_features_or_bias)

        in_feat = int(in_features if in_features is not None else 64)
        out_feat = int(out_features if out_features is not None else in_feat)
        self.in_features = in_feat
        self.out_features = out_feat

        eff_rank = rank if rank is not None else (svd_rank if svd_rank is not None else kwargs.get("r", 128))
        self.energy = float(energy)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)

        resolved_rank = max(1, min(int(eff_rank), self.in_features, self.out_features))
        if self.max_rank > 0:
            resolved_rank = min(resolved_rank, self.max_rank)
        resolved_rank = max(resolved_rank, min(self.min_rank, self.in_features, self.out_features))
        self.rank = resolved_rank
        self.svd_rank = resolved_rank

        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        factory_kwargs = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)
        self.lin1 = self.down
        self.lin2 = self.up

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

        self.reset_parameters()

        if freeze_base:
            self.freeze_base_weights()

        if isinstance(in_features_or_weight, torch.Tensor):
            b = out_features_or_bias if isinstance(out_features_or_bias, torch.Tensor) else None
            self.decompose_from_weight(in_features_or_weight, b)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            if self.up.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.up.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.up.bias, -bound, bound)
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if self.lora_B is not None:
                nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    @property
    def U(self) -> nn.Parameter:
        return self.up.weight

    @property
    def V(self) -> torch.Tensor:
        return self.down.weight.t()

    @property
    def S(self) -> torch.Tensor:
        return torch.ones(self.rank, device=self.up.weight.device, dtype=self.up.weight.dtype)

    @property
    def A(self) -> nn.Parameter | None:
        return self.lora_B

    @property
    def B(self) -> nn.Parameter | None:
        return self.lora_A

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return w

    @property
    def base_weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> nn.Parameter | None:
        return self.up.bias

    def effective_weight(self) -> torch.Tensor:
        return self.weight

    def effective_bias(self) -> nn.Parameter | None:
        return self.up.bias

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        energy: float | None = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            dev = W.device
            W_cpu = W.detach().cpu().float()
            U_cpu, S_cpu, Vt_cpu = torch.linalg.svd(W_cpu, full_matrices=False)
            U = U_cpu.to(dev)
            S = S_cpu.to(dev)
            Vt = Vt_cpu.to(dev)

            target_energy = energy if energy is not None else self.energy
            total_energy = (S**2).sum()
            cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)
            calc_rank = int((cum_energy < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            sqrt_s = torch.sqrt(S[:r])
            self.up.weight.data.zero_()
            self.up.weight.data[:, :r] = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
            self.down.weight.data.zero_()
            self.down.weight.data[:r, :] = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(
                self.down.weight.dtype
            )

            if bias is not None and self.up.bias is not None:
                min_b = min(self.up.bias.numel(), bias.numel())
                self.up.bias.data.zero_()
                self.up.bias.data[:min_b].copy_(bias[:min_b].to(self.up.bias.dtype))

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
    ) -> FactorizedLinear:
        """Construct a FactorizedLinear initialized from an nn.Linear layer via SVD."""
        mod = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            rank=rank,
            lora_rank=lora_rank,
            bias=bias and (layer.bias is not None),
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )
        mod.decompose_from_weight(layer.weight, layer.bias)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x, self.lora_A.to(dtype=x.dtype)),
                self.lora_B.to(dtype=x.dtype),
            ) * self.lora_scaling
            out = out + lora_out
        return out


FactorisedLinear = FactorizedLinear

__all__ = ["FactorizedLinear", "FactorisedLinear"]
