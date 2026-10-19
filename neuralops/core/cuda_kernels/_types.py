# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
 Types.Py

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import torch


class DataType(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class KernelLaunchConfig:
    """CUDA kernel execution configuration."""
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    shared_mem_bytes: int = 0
    stream: torch.cuda.Stream | None = None
