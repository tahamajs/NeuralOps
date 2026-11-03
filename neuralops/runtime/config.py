#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Runtime Configuration Protocol and Hardware Device Resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import torch


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration specifying hardware devices, precision, and logging.

    Attributes:
        device: Target execution device ("auto", "cuda", "cpu", "mps", or explicit e.g. "cuda:0").
        dtype: Numerical precision ("auto", "float16", "bfloat16", "float32").
        seed: Random seed for deterministic weight generation and shuffling.
        deterministic: Whether to enforce deterministic CUDA convolution and algorithms.
        log_level: Standard logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        cache_dir: Optional path for temporary model weights and Triton/CUDA caches.
        max_memory_fraction: VRAM fraction threshold before throwing resource errors.
    """

    device: str = "auto"
    dtype: str = "auto"
    seed: int = 42
    deterministic: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cache_dir: Optional[str] = None
    max_memory_fraction: float = 0.90

    def resolve_device(self) -> torch.device:
        """Resolves the active torch.device based on available accelerators."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        """Resolves the torch.dtype based on precision settings and hardware support."""
        if self.dtype == "auto":
            if torch.cuda.is_available():
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if self.dtype in mapping:
            return mapping[self.dtype]
        raise ValueError(
            f"Unsupported dtype '{self.dtype}'. Supported values: {list(mapping.keys())}"
        )
