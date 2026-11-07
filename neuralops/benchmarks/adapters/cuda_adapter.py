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
benchmark execution framework. Implements the 6-gate lifecycle inheriting from
BaseBenchmarkAdapter, the 3-level fallback hierarchy
(Level 1 Native CUDA C++ -> Level 2 Triton JIT -> Level 3 Vectorized PyTorch reference),
zero silent fallback logging, and hardware capability inspection across all canonical
CUDA modules and native CUDA C++ kernel implementations.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter, BenchmarkInputs, NumericsReport
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.ops.cuda import list_cuda_kernels
from neuralops.ops.cuda._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    get_last_dispatch_backend,
    get_last_dispatch_error,
)

logger = logging.getLogger(__name__)

# The unique canonical CUDA C++ kernel module package names
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


class _DeviceStr(str):
    """String subclass providing .type property for torch.device compatibility."""

    @property
    def type(self) -> str:
        return torch.device(self).type


class CUDAKernelAdapter(BaseBenchmarkAdapter):
    """Adapter bridging native CUDA C++ kernels to benchmark pipelines.

    Adheres strictly to the 3-level fallback contract:
      - Level 1: Native compiled CUDA C++ extension
      - Level 2: Triton JIT autotuned kernel
      - Level 3: Vectorized PyTorch CPU/MPS reference

    Inherits from BaseBenchmarkAdapter and implements the complete 6-gate
    lifecycle returning canonical BenchmarkResult instances.
    """

    def __init__(
        self,
        kernel_name: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        tolerance: float = BaseBenchmarkAdapter.DEFAULT_TOLERANCE,
        timeout_s: float = BaseBenchmarkAdapter.DEFAULT_TIMEOUT_S,
        warmup_steps: int = 5,
        measure_steps: int = 20,
    ) -> None:
        dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        dev_obj = _DeviceStr(dev_str)
        super().__init__(
            benchmark_name=kernel_name,
            device=dev_obj,
            dtype=dtype,
            seed=seed,
            tolerance=tolerance,
            timeout_s=timeout_s,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )
        self.kernel_name = kernel_name
        self.device = dev_obj
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
                for cand in (
                    mod_name,
                    base_name,
                    f"{mod_name}_cuda",
                    f"{mod_name}_fwd",
                    f"{mod_name}_forward",
                    "forward",
                    "fwd",
                ):
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
        """Execute kernel via 3-tier fallback or run full 6-gate benchmark lifecycle.

        If called with input arguments, behaves as direct kernel invocation (backward-compatible).
        If called without arguments, runs the complete BaseBenchmarkAdapter 6-gate lifecycle
        and returns a canonical BenchmarkResult.
        """
        if args or kwargs:
            return self.dispatch_kernel(*args, **kwargs)
        return super().execute()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.dispatch_kernel(*args, **kwargs)

    # ------------------------------------------------------------------
    # 6-Gate Benchmark Lifecycle Implementation
    # ------------------------------------------------------------------

    def capability_check(self) -> Dict[str, Any]:
        """Gate 1: probe environment only — NO kernel execution."""
        is_cuda_req = str(self.device).startswith("cuda")
        if is_cuda_req and not torch.cuda.is_available():
            return {
                "available": False,
                "reason": f"Device '{self.device}' requested but CUDA is not available on this host",
                "backend": "none",
            }

        if is_cuda_req and torch.cuda.is_available() and self._cuda_fn is not None:
            return {
                "available": True,
                "reason": "",
                "backend": "cuda",
            }

        if is_cuda_req and torch.cuda.is_available() and self._triton_fn is not None:
            return {
                "available": True,
                "reason": "Native CUDA unavailable; falling back to Triton JIT",
                "backend": "triton",
            }

        if self._pytorch_ref_fn is not None:
            reason = "" if not is_cuda_req else "CUDA/Triton unavailable; falling back to PyTorch reference"
            return {
                "available": True,
                "reason": reason,
                "backend": "pytorch",
            }

        return {
            "available": False,
            "reason": f"No native CUDA, Triton, or PyTorch reference resolved for '{self.kernel_name}'",
            "backend": "none",
        }

    def build_inputs(self) -> BenchmarkInputs:
        """Gate 2: construct deterministic input tensors."""
        torch.manual_seed(self.seed)
        dtype = self.dtype
        dev = self.device
        name = self.kernel_name.lower()

        if any(k in name for k in ("cosformer", "linear", "attention", "attn", "gqa", "sliding_window")):
            q = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            k = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            v = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            tensors = {"q": q, "k": k, "v": v}
            args: Tuple[Any, ...] = (q, k, v)
            kwargs: Dict[str, Any] = {}
        elif any(k in name for k in ("moe", "routing", "router", "expert")):
            x = torch.randn(4, 32, device=dev, dtype=dtype)
            w = torch.randn(32, 4, device=dev, dtype=dtype)
            tensors = {"x": x, "w": w}
            args = (x, w)
            kwargs = {}
        elif "layernorm" in name or "norm" in name:
            x = torch.randn(2, 4, 16, device=dev, dtype=dtype)
            w = torch.ones(16, device=dev, dtype=dtype)
            b = torch.zeros(16, device=dev, dtype=dtype)
            tensors = {"x": x, "weight": w, "bias": b}
            args = (x, w, b)
            kwargs = {}
        elif "gemm" in name:
            x = torch.randn(16, 32, device=dev, dtype=dtype)
            w = torch.randn(32, 16, device=dev, dtype=dtype)
            tensors = {"x": x, "w": w}
            args = (x, w)
            kwargs = {}
        elif "scan" in name:
            u = torch.randn(2, 8, 16, device=dev, dtype=dtype)
            delta = torch.nn.functional.softplus(torch.randn(2, 8, 16, device=dev, dtype=dtype))
            a = -torch.abs(torch.randn(8, 4, device=dev, dtype=dtype)) - 0.1
            b_t = torch.randn(2, 4, 16, device=dev, dtype=dtype)
            c = torch.randn(2, 4, 16, device=dev, dtype=dtype)
            d = torch.randn(8, device=dev, dtype=dtype)
            tensors = {"u": u, "delta": delta, "a": a, "b": b_t, "c": c, "d": d}
            args = (u, delta, a, b_t, c, d)
            kwargs = {}
        else:
            x = torch.randn(2, 16, 32, device=dev, dtype=dtype)
            tensors = {"x": x}
            args = (x,)
            kwargs = {}

        return BenchmarkInputs(
            tensors=tensors,
            metadata={"args": args, "kwargs": kwargs},
            seed=self.seed,
            dtype=self.dtype,
            device=str(self.device),
        )

    def run_reference(self, inputs: BenchmarkInputs) -> torch.Tensor:
        """Gate 3: run reference implementation (vectorized PyTorch)."""
        args = inputs.metadata.get("args")
        kwargs = inputs.metadata.get("kwargs", {})
        if args is None:
            args = tuple(inputs.tensors.values())

        if self._pytorch_ref_fn is not None:
            out = self._pytorch_ref_fn(*args, **kwargs)
        else:
            raise RuntimeError(f"No PyTorch reference resolved for kernel '{self.kernel_name}'")

        if isinstance(out, tuple):
            out = out[0]
        return out

    def run_candidate(self, inputs: BenchmarkInputs) -> Tuple[torch.Tensor, str]:
        """Gate 4: run candidate backend."""
        args = inputs.metadata.get("args")
        kwargs = inputs.metadata.get("kwargs", {})
        if args is None:
            args = tuple(inputs.tensors.values())

        is_cuda_req = str(self.device).startswith("cuda") and torch.cuda.is_available()

        if is_cuda_req and self._cuda_fn is not None:
            out = self._cuda_fn(*args, **kwargs)
            backend_used = "cuda"
        elif is_cuda_req and self._triton_fn is not None:
            out = self._triton_fn(*args, **kwargs)
            backend_used = "triton"
        elif self._pytorch_ref_fn is not None:
            out = self._pytorch_ref_fn(*args, **kwargs)
            backend_used = "pytorch"
        else:
            out = self.dispatch_kernel(*args, **kwargs)
            backend_used = get_last_dispatch_backend(self.kernel_name)
            if backend_used == "unknown":
                backend_used = "cuda" if is_cuda_req else "pytorch"

        if isinstance(out, tuple):
            out = out[0]
        return out, backend_used

    def validate_numerics(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> NumericsReport:
        """Gate 5: compare reference vs candidate numerically."""
        return self._default_validate_numerics(reference, candidate)

    def measure_performance(self, inputs: BenchmarkInputs) -> Dict[str, float]:
        """Gate 6: timing ONLY after Gate 5 passed."""
        args = inputs.metadata.get("args")
        kwargs = inputs.metadata.get("kwargs", {})
        if args is None:
            args = tuple(inputs.tensors.values())

        is_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()

        for _ in range(self.warmup_steps):
            _ = self.dispatch_kernel(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()

        latencies: List[float] = []
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(self.measure_steps):
                start_event.record()
                _ = self.dispatch_kernel(*args, **kwargs)
                end_event.record()
                end_event.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            for _ in range(self.measure_steps):
                t0 = time.perf_counter()
                _ = self.dispatch_kernel(*args, **kwargs)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
            peak_mem_mb = 0.0

        lat_sorted = sorted(latencies)
        n = len(lat_sorted)
        mean_lat = sum(lat_sorted) / max(n, 1)
        variance = sum((x - mean_lat) ** 2 for x in lat_sorted) / max(n - 1, 1) if n > 1 else 0.0
        std_lat = math.sqrt(variance)
        p50 = lat_sorted[int(n * 0.50)] if n else 0.0
        p90 = lat_sorted[min(int(n * 0.90), n - 1)] if n else 0.0
        p99 = lat_sorted[min(int(n * 0.99), n - 1)] if n else 0.0

        return {
            "latency_ms_mean": mean_lat,
            "latency_ms_std": std_lat,
            "latency_ms_p50": p50,
            "latency_ms_p90": p90,
            "latency_ms_p99": p99,
            "memory_mb_peak": peak_mem_mb,
        }

    # ------------------------------------------------------------------
    # Module & Kernel Catalogs
    # ------------------------------------------------------------------

    @staticmethod
    def list_modules() -> List[str]:
        """Return the unique CUDA module package names."""
        return sorted(list(CANONICAL_CUDA_MODULES))

    @staticmethod
    def list_all_modules() -> List[str]:
        """Return the unique CUDA module package names."""
        return sorted(list(CANONICAL_CUDA_MODULES))

    @staticmethod
    def list_all_kernels() -> List[str]:
        """Return all native CUDA kernel implementations."""
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
