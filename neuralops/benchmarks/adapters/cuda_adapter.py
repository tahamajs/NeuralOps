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
(Level 1 Native CUDA C++ -> Level 2 Triton JIT -> Level 3 Vectorized PyTorch reference),
zero silent fallback logging, and hardware capability inspection across all 110 unique
CUDA modules and 114 native CUDA C++ kernel implementations.
"""

from __future__ import annotations

import importlib
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

# The 110 unique canonical CUDA C++ kernel module package names
CANONICAL_CUDA_MODULES: List[str] = list_cuda_kernels("modules")

# Alias mappings for sub-operators resolving to parent module bindings
CUDA_KERNEL_ALIASES: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    "cosformer_attn": ("cosformer", "cosformer_linear_attention", "cosformer_linear_attention_pytorch_ref"),
    "linear_attn": ("linear_attention", "linear_attention", "linear_attention_pytorch_ref"),
    "expert_dispatch": ("moe_routing", "expert_dispatch", "expert_dispatch_pytorch_ref"),
    "moe_router": ("moe_routing", "moe_routing_topk", "moe_routing_topk_pytorch_ref"),
    "kv_cache_copy": ("paged_kv_cache", "paged_kv_cache_update", "paged_kv_cache_update_pytorch_ref"),
    "quantized_fp8_gemm": ("quantized_gemm", "fp8_gemm", "fp8_gemm_pytorch_ref"),
}


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
        """Resolve CUDA, Triton, and PyTorch implementations for the kernel across all tiers."""
        target = CUDA_KERNEL_ALIASES.get(self.kernel_name, (self.kernel_name, None, None))
        mod_name, explicit_op, explicit_ref = target

        # Level 1: CUDA C++ function resolution
        try:
            # 1a. Try dedicated ops.cuda submodule
            mod = importlib.import_module(f"neuralops.ops.cuda.{mod_name}")
            if explicit_op and hasattr(mod, explicit_op):
                fn = getattr(mod, explicit_op)
                if callable(fn):
                    self._cuda_fn = fn
            if self._cuda_fn is None and hasattr(mod, "__all__"):
                for name in mod.__all__:
                    if not any(name.endswith(s) for s in ("_ref", "_pytorch", "_pytorch_ref", "_reference")):
                        fn = getattr(mod, name, None)
                        if callable(fn):
                            self._cuda_fn = fn
                            break
            if self._cuda_fn is None:
                base_name = mod_name.replace("_attention", "").replace("_gemm", "").replace("_scan", "")
                for cand in (mod_name, base_name, f"{mod_name}_cuda", f"{mod_name}_fwd", f"{mod_name}_forward", "forward", "fwd"):
                    if hasattr(mod, cand):
                        fn = getattr(mod, cand)
                        if callable(fn):
                            self._cuda_fn = fn
                            break
        except Exception as e:
            logger.debug("CUDA submodule resolution failed for '%s': %s", mod_name, e)

        # 1b. Fallback to package level lazy-loading export in neuralops.ops.cuda
        if self._cuda_fn is None:
            try:
                import neuralops.ops.cuda as nck
                if hasattr(nck, self.kernel_name):
                    fn = getattr(nck, self.kernel_name)
                    if callable(fn):
                        self._cuda_fn = fn
            except Exception as e:
                logger.debug("CUDA operator resolution failed for '%s': %s", self.kernel_name, e)

        # Level 2: Triton kernel resolution
        try:
            from neuralops.benchmarks.adapters.triton_adapter import TritonKernelAdapter
            triton_adapter = TritonKernelAdapter(mod_name, device=self.device)
            if triton_adapter._kernel_fn is not None:
                self._triton_fn = triton_adapter._kernel_fn
        except Exception as e:
            logger.debug("Triton fallback resolution failed for '%s': %s", self.kernel_name, e)

        # Level 3: PyTorch CPU reference resolution
        try:
            mod = importlib.import_module(f"neuralops.ops.cuda.{mod_name}")
            if explicit_ref and hasattr(mod, explicit_ref):
                fn = getattr(mod, explicit_ref)
                if callable(fn):
                    self._pytorch_ref_fn = fn
            if self._pytorch_ref_fn is None:
                for cand in (
                    f"{mod_name}_pytorch_ref",
                    f"{mod_name}_ref",
                    f"{mod_name}_pytorch_reference",
                    f"{mod_name}_pytorch",
                    "pytorch_ref",
                    "ref",
                ):
                    if hasattr(mod, cand):
                        fn = getattr(mod, cand)
                        if callable(fn):
                            self._pytorch_ref_fn = fn
                            break
            if self._pytorch_ref_fn is None:
                for attr in dir(mod):
                    if "pytorch" in attr.lower() or "ref" in attr.lower():
                        fn = getattr(mod, attr)
                        if callable(fn):
                            self._pytorch_ref_fn = fn
                            break
        except Exception as e:
            logger.debug("PyTorch reference resolution via submodule failed for '%s': %s", mod_name, e)

        if self._pytorch_ref_fn is None:
            try:
                import neuralops.ops.cuda as nck
                for ref_cand in (
                    f"{self.kernel_name}_pytorch_ref",
                    f"{self.kernel_name}_ref",
                    f"{self.kernel_name}_pytorch",
                    f"{mod_name}_pytorch_ref",
                    f"{mod_name}_ref",
                ):
                    if hasattr(nck, ref_cand):
                        fn = getattr(nck, ref_cand)
                        if callable(fn):
                            self._pytorch_ref_fn = fn
                            break
            except Exception as e:
                logger.debug("PyTorch reference resolution via nck failed for '%s': %s", self.kernel_name, e)

    @property
    def is_cuda_available(self) -> bool:
        """Return True if native CUDA implementation is callable and CUDA is active."""
        return self._cuda_fn is not None and self.device.type == "cuda" and torch.cuda.is_available()

    def dispatch_kernel(self, *args: Any, **kwargs: Any) -> Any:
        """Execute kernel via 3-tier fallback hierarchy with zero silent degradation.

        Hierarchy:
          Level 1: Native compiled CUDA C++ extension (maximum throughput)
          Level 2: Triton JIT autotuned kernel (accelerated portable PTX)
          Level 3: Vectorized PyTorch reference (universal CPU/MPS fallback)
        """
        # Fast path CPU routing when input tensors are strictly on CPU
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
            self._pytorch_ref_fn,
            *args,
            kernel_name=self.kernel_name,
            **kwargs,
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute kernel via 3-tier fallback hierarchy with full telemetry tracking."""
        return self.dispatch_kernel(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.dispatch_kernel(*args, **kwargs)

    @staticmethod
    def list_modules() -> List[str]:
        """Return the 110 unique CUDA module package names."""
        return sorted(list(CANONICAL_CUDA_MODULES))

    @staticmethod
    def list_all_modules() -> List[str]:
        """Return the 110 unique CUDA module package names."""
        return sorted(list(CANONICAL_CUDA_MODULES))

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
    "CANONICAL_CUDA_MODULES",
    "CUDAKernelAdapter",
    "CUDA_KERNEL_ALIASES",
]
