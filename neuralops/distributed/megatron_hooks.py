#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Megatron-LM Tensor Parallel Hooks for Low-Rank Factored Linear Layers

Implements ColumnParallelLinear and RowParallelLinear modules supporting
factorized low-rank matrix projections partitioned across tensor parallel ranks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
]


class ColumnParallelLinear(nn.Module):
    """Linear layer partitioned along output features (columns) across tensor-parallel ranks.

    Compatible with Megatron-LM tensor model parallelism and low-rank factorization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        rank: Optional[int] = None,
        tp_world_size: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tp_world_size = tp_world_size
        self.split_out_features = out_features // tp_world_size

        if rank is not None:
            self.low_rank = rank
            self.linear_in = nn.Linear(in_features, rank, bias=False)
            self.linear_out = nn.Linear(rank, self.split_out_features, bias=bias)
        else:
            self.low_rank = None
            self.linear = nn.Linear(in_features, self.split_out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank is not None:
            intermediate = self.linear_in(x)
            out = self.linear_out(intermediate)
        else:
            out = self.linear(x)

        if self.gather_output and self.tp_world_size > 1 and torch.distributed.is_initialized():
            outputs = [torch.empty_like(out) for _ in range(self.tp_world_size)]
            torch.distributed.all_gather(outputs, out)
            return torch.cat(outputs, dim=-1)

        return out


class RowParallelLinear(nn.Module):
    """Linear layer partitioned along input features (rows) across tensor-parallel ranks.

    Compatible with Megatron-LM tensor model parallelism and low-rank factorization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        rank: Optional[int] = None,
        tp_world_size: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.tp_world_size = tp_world_size
        self.split_in_features = in_features // tp_world_size

        if rank is not None:
            self.low_rank = rank
            self.linear_in = nn.Linear(self.split_in_features, rank, bias=False)
            self.linear_out = nn.Linear(rank, out_features, bias=bias)
        else:
            self.low_rank = None
            self.linear = nn.Linear(self.split_in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank is not None:
            intermediate = self.linear_in(x)
            out = self.linear_out(intermediate)
        else:
            out = self.linear(x)

        if self.tp_world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.all_reduce(out)

        return out
