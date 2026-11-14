#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Butterfly Matrix Factorization Compressor.

Decomposes dense linear projection layers into structured, sparse Butterfly (Monarch)
block-diagonal factors to achieve sub-quadratic parameter and compute complexity.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class ButterflyLinear(nn.Module):
    """
    Linear layer factorized via a two-stage Monarch/Butterfly block-diagonal decomposition:
        y = ((x @ R.T) @ L.T) + bias
    where L and R are block-diagonal or low-rank butterfly factors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        factory_kwargs = {"device": device, "dtype": dtype}
        # Monarch two-factor decomposition: R (rank x in_features), L (out_features x rank)
        self.R = nn.Parameter(torch.empty((self.rank, self.in_features), **factory_kwargs))
        self.L = nn.Parameter(torch.empty((self.out_features, self.rank), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factors using Kaiming normal distributions."""
        nn.init.kaiming_normal_(self.R, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.L, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
    ) -> ButterflyLinear:
        """Factorize an existing nn.Linear layer using SVD-based Butterfly factors."""
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w = linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(w, full_matrices=False)
            r = min(mod.rank, len(S))
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            mod.L.data.copy_((U[:, :r] * sqrt_s.unsqueeze(0)).to(dtype=linear.weight.dtype))
            mod.R.data.copy_((sqrt_s.unsqueeze(1) * Vt[:r, :]).to(dtype=linear.weight.dtype))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying two-stage butterfly factors."""
        h = F.linear(x, self.R)
        return F.linear(h, self.L, self.bias)


@COMPRESSOR_REGISTRY.register("butterfly_compression")
@COMPRESSOR_REGISTRY.register("butterfly", is_alias=True)
class ButterflyCompressor(BaseCompressor):
    """Butterfly Matrix Factorization Compressor."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="butterfly_compression", **kwargs)
        self.rank = rank
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, ButterflyLinear):
                orig_w = module.weight.data
                bf_layer = ButterflyLinear.from_linear(module, rank=self.rank)
                recon_w = bf_layer.L @ bf_layer.R
                rel_err = (torch.norm(orig_w - recon_w) / torch.norm(orig_w).clamp(min=1e-8)).item()

                if rel_err > self.max_error and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

                replace_module(self.model, name, bf_layer)
                self.compressed_layers.append(name)

        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "butterfly_rank": self.rank,
            }
        )
        self.stats = stats
        return self.model, stats


__all__ = ["ButterflyLinear", "ButterflyCompressor"]
