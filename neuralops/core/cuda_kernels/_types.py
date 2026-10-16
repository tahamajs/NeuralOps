"""Shared type definitions and launch configurations for CUDA C++ hotpaths."""

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
