#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUDA C++ Kernel Benchmark Adapter.

Bridges native CUDA C++ kernel bindings (Tier 1) in neuralops/ops/cuda/ to the
benchmark execution framework. Implements the 3-level fallback hierarchy
(Native CUDA C++ -> Triton JIT -> Vectorized PyTorch reference), zero silent
fallback logging, and hardware capability inspection.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.ops.cuda import list_cuda_kernels
from neuralops.ops.cuda._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    get_last_dispatch_backend,
    get_last_dispatch_error,
)

logger = logging.getLogger(__name__)


class CUDAKernelAdapter:
    """Adapter bridging native CUDA C++ kernels to benchmark pipelines.
    
    Adheres strictly to the 3-level fallback contract:
      - Level 1: Native compiled CUDA C++ extension
      - Level 2: Triton JIT autotuned kernel
      - Level 3: Vectorized PyTorch CPU/MPS reference
    """

    def __init__(
        self,
        kernel_name: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.kernel_name = kernel_name
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cuda_fn: Optional[Callable[..., Any]] = None
        self._triton_fn: Optional[Callable[..., Any]] = None
        self._pytorch_ref_fn: Optional[Callable[..., Any]] = None
        self._resolve_kernel()

    def _resolve_kernel(self) -> None:
        """Resolve CUDA, Triton, and PyTorch implementations for the kernel."""
        # Level 1: CUDA C++ function
        try:
            import neuralops.ops.cuda as nck
            if hasattr(nck, self.kernel_name):
                fn = getattr(nck, self.kernel_name)
                if callable(fn):
                    self._cuda_fn = fn
        except Exception as e:
            logger.debug("CUDA operator resolution failed for '%s': %s", self.kernel_name, e)

        # Level 2: Triton kernel
        try:
            from neuralops.benchmarks.adapters.triton_adapter import TritonKernelAdapter
            triton_adapter = TritonKernelAdapter(self.kernel_name, device=self.device)
            if triton_adapter._kernel_fn is not None:
                self._triton_fn = triton_adapter._kernel_fn
        except Exception as e:
            logger.debug("Triton fallback resolution failed for '%s': %s", self.kernel_name, e)

        # Level 3: PyTorch CPU reference
        try:
            from neuralops.ops.cuda._bindings import _get_pytorch_reference
            ref = _get_pytorch_reference(self.kernel_name)
            if ref is not None and callable(ref):
                self._pytorch_ref_fn = ref
        except Exception as e:
            logger.debug("PyTorch reference resolution failed for '%s': %s", self.kernel_name, e)

    @property
    def is_cuda_available(self) -> bool:
        """Return True if native CUDA implementation is callable and CUDA is active."""
        return self._cuda_fn is not None and self.device.type == "cuda" and torch.cuda.is_available()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute kernel via 3-level dispatch hierarchy with full telemetry tracking."""
        # Handle CPU tensors or non-CUDA environments gracefully
        first_tensor_cpu = False
        for a in args:
            if isinstance(a, torch.Tensor):
                first_tensor_cpu = not a.is_cuda
                break
        if not first_tensor_cpu:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    first_tensor_cpu = not v.is_cuda
                    break

        if first_tensor_cpu or not torch.cuda.is_available() or self.device.type != "cuda":
            if self._pytorch_ref_fn is not None:
                return self._pytorch_ref_fn(*args, **kwargs)

        return dispatch(
            self._cuda_fn,
            self._triton_fn,
            pytorch_ref=self._pytorch_ref_fn,
            kernel_name=self.kernel_name,
            *args,
            **kwargs,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.execute(*args, **kwargs)

    @staticmethod
    def list_all_modules() -> List[str]:
        """Return the 110 unique CUDA module package names."""
        return list_cuda_kernels("modules")

    @staticmethod
    def list_all_kernels() -> List[str]:
        """Return all 114 native CUDA kernel implementations."""
        return list_cuda_kernels("all")

    @staticmethod
    def get_dispatch_telemetry() -> Dict[str, Any]:
        """Return global dispatch telemetry metrics."""
        return {
            "stats": get_dispatch_stats(),
        }


__all__ = [
    "CUDAKernelAdapter",
]
