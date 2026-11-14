#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
LoKr (Low-Rank Kronecker Product) Compressor.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.peft.lokr import LoKrLinear
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("lokr")
@COMPRESSOR_REGISTRY.register("lokr_compressor", is_alias=True)
class LoKrCompressor(BaseCompressor):
    """Compresses/Adapts linear layers using Low-Rank Kronecker Product (LoKr)."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lokr", **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.compressed_layers: list[str] = []

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, LoKrLinear):
                lokr_layer = LoKrLinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                )
                replace_module(self.model, name, lokr_layer)
                self.compressed_layers.append(name)

        comp_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "num_compressed": len(self.compressed_layers),
                "lokr_rank": self.rank,
                "lokr_alpha": self.alpha,
            }
        )
        self.stats = stats
        return self.model, stats


__all__ = ["LoKrLinear", "LoKrCompressor"]
