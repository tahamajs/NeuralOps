# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Component layers and config for ORION Attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ORIONConfig:
    svd_energy: float = 0.95
    svd_min_rank: int = 4
    svd_max_rank: int = 64
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    rff_features: int = 256
    softpick_temperature: float = 1.0
    dropout: float = 0.1


class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.energy = energy
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        rank = max(min_rank, min(max_rank, min(in_features, out_features)))
        self.rank = rank
        self.svd_rank = rank

        factory_kwargs = {"device": device, "dtype": dtype}
        self.down = nn.Linear(in_features, rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(rank, out_features, bias=bias, **factory_kwargs)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        out_dim, in_dim = weight.shape
        with torch.no_grad():
            W = weight.float()
            if W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / torch.clamp(total, min=1e-8)
            rank = int((cum < self.energy).sum().item()) + 1
            max_avail_rank = self.down.weight.shape[0]
            rank = max(
                self.min_rank,
                min(
                    rank,
                    min(self.out_features, self.in_features),
                    self.max_rank,
                    max_avail_rank,
                ),
            )
            self.rank = rank

            device = self.down.weight.device
            dtype = self.down.weight.dtype

            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:rank, : self.in_features].copy_(B.to(device=device, dtype=dtype))
            self.up.weight.data[: self.out_features, :rank].copy_(A.to(device=device, dtype=dtype))

            if bias is not None and self.up.bias is not None:
                min_b = min(self.up.bias.numel(), bias.numel())
                self.up.bias.data[:min_b].copy_(bias[:min_b].to(device=device, dtype=dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            if self.freeze_base:
                self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            try:
                from neuralops.ops.cuda.low_rank_svd import fused_lowrank_gated_gemm

                base_out = fused_lowrank_gated_gemm(
                    x, self.down.weight, self.up.weight.t().contiguous(), bias=self.up.bias
                )
                if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                    lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
                    return base_out + lora_out
                return base_out
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass
            try:
                from neuralops.ops.triton._types import FORCE_PYTORCH_FALLBACK

                if not FORCE_PYTORCH_FALLBACK:
                    from neuralops.ops.triton.low_rank import fused_down_up_gemm_bias_triton

                    base_out = fused_down_up_gemm_bias_triton(
                        x, self.down.weight.t().contiguous(), self.up.weight.t().contiguous(), self.up.bias
                    )
                    if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
                        return base_out + lora_out
                    return base_out
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out



__all__ = [
    "ORIONConfig",
    "FactorisedLinear",
]
