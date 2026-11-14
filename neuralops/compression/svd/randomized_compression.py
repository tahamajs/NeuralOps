#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Randomized SVD (rSVD) Compressor.

Decomposes linear projection matrices into low-rank factors using randomized Gaussian
subspace projections and normalized power iterations (Halko et al., 2011).
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
from neuralops.core.svd_utils import randomized_svd

logger = logging.getLogger(__name__)


class RandomizedSVDLinear(nn.Module):
    """
    Linear layer decomposed into two low-rank factors via Randomized SVD:
        y = ((x @ W1.T) @ W2.T) + bias
    where W1 is (rank x in_features) and W2 is (out_features x rank).
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
        self.W1 = nn.Parameter(torch.empty((self.rank, self.in_features), **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty((self.out_features, self.rank), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        n_iter: int = 4,
    ) -> RandomizedSVDLinear:
        """Factorize an existing linear layer using randomized SVD."""
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w = linear.weight.data
            U, S, Vt = randomized_svd(w, rank=mod.rank, n_iter=n_iter)
            sqrt_s = torch.sqrt(torch.clamp(S, min=1e-12))
            mod.W2.data.copy_((U * sqrt_s.unsqueeze(0)).to(dtype=linear.weight.dtype))
            mod.W1.data.copy_((sqrt_s.unsqueeze(1) * Vt).to(dtype=linear.weight.dtype))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(linear.bias.data)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using factored low-rank projection."""
        h = F.linear(x, self.W1)
        return F.linear(h, self.W2, self.bias)


@COMPRESSOR_REGISTRY.register("randomized_svd_compressor")
@COMPRESSOR_REGISTRY.register("randomized_svd", is_alias=True)
@COMPRESSOR_REGISTRY.register("randomized_compression", is_alias=True)
@COMPRESSOR_REGISTRY.register("rsvd_compressor", is_alias=True)
class RandomizedSVDCompressor(BaseCompressor):
    """Randomized Singular Value Decomposition (rSVD) Compressor."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        n_iter: int = 4,
        max_error: float = 0.60,
        fallback_to_original: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="randomized_svd_compressor", **kwargs)
        self.rank = rank
        self.n_iter = n_iter
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, RandomizedSVDLinear):
                orig_w = module.weight.data
                rsvd_layer = RandomizedSVDLinear.from_linear(module, rank=self.rank, n_iter=self.n_iter)
                recon_w = rsvd_layer.W2 @ rsvd_layer.W1
                rel_err = (torch.norm(orig_w - recon_w) / torch.norm(orig_w).clamp(min=1e-8)).item()

                if rel_err > self.max_error and self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

                replace_module(self.model, name, rsvd_layer)
                self.compressed_layers.append(name)

        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "rsvd_rank": self.rank,
            "n_iter": self.n_iter,
        })
        self.stats = stats
        return self.model, stats


__all__ = ["RandomizedSVDLinear", "RandomizedSVDCompressor"]
