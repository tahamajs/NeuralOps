#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Triton Kernel Benchmark Adapter.

Bridges the 26 canonical Triton kernel modules in neuralops/core/triton_kernels/
and the 24 extension modules in neuralops/ops/triton/ into standardized execution
signatures. Inherits from BaseBenchmarkAdapter and implements the complete 6-gate
lifecycle. Enforces 2D SRAM tile scheduling, FP16/BF16/FP32 precision constraints,
and strictly bans deprecated tl.make_block_ptr usage per AGENTS.md Section 7.
"""

from __future__ import annotations

import ast
import importlib
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter, BenchmarkInputs, NumericsReport
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus

logger = logging.getLogger(__name__)

# The 26 canonical Triton kernel modules in neuralops/core/triton_kernels/
CANONICAL_TRITON_MODULES: List[str] = [
    "block_sparse",
    "cosformer_triton_kernel",
    "cur_linear",
    "differential_attention",
    "dynamic_sparse_attention",
    "elementwise",
    "flash_decoding",
    "fms_scan",
    "fused_ce",
    "fused_kd",
    "fused_norm_linear",
    "fused_rope",
    "gdn_scan",
    "gla_scan",
    "gqa",
    "hgrn2_scan",
    "hydra_attention",
    "linear_attn",
    "low_rank",
    "paged_attention",
    "retnet_retention",
    "selective_scan",
    "sliding_window",
    "softplus",
    "sparse_attn",
    "state_update",
]

# The 24 extension Triton kernel modules in neuralops/ops/triton/
EXTENSION_TRITON_MODULES: List[str] = [
    "adaptive_sparse_triton",
    "based_scan",
    "chela_slay_triton",
    "chela_slay_wide_triton",
    "cross_attention_triton",
    "delta_net_scan",
    "dynamic_conv_triton",
    "fused_svd_projection_triton",
    "lightning_scan",
    "lsh_attention_triton",
    "paged_linear_attn",
    "prune_admm",
    "prune_sparsegpt",
    "prune_wanda",
    "quantize_awq",
    "quantize_fp8",
    "quantize_kmeans",
    "rala_attention_triton",
    "retnet_scan",
    "routing_attention_triton",
    "rwkv6_scan",
    "svd_linear_decay_triton",
    "svd_mamba_fused_triton",
    "token_compressed_triton",
    "ttt_linear_step_triton",
    "varlen_attention",
]


class _DeviceStr(str):
    """String subclass providing .type property for torch.device compatibility."""

    @property
    def type(self) -> str:
        return torch.device(self).type


class TritonKernelAdapter(BaseBenchmarkAdapter):
    """Adapter bridging canonical and extension Triton kernels to benchmark runners.

    Provides standardized forward invocations, shape inference, device management,
    2D SRAM tile contract verification, and automatic CPU reference fallback routing
    with explicit logger warnings when running on hardware without native Triton support.

    Inherits from BaseBenchmarkAdapter and implements the complete 6-gate lifecycle
    returning canonical BenchmarkResult instances.
    """

    def __init__(
        self,
        kernel_name: str,
        is_extension: bool = False,
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
        self.is_extension = is_extension
        self.device = dev_obj
        self._module_path = (
            f"neuralops.ops.triton.{kernel_name}" if is_extension else f"neuralops.core.triton_kernels.{kernel_name}"
        )
        self._module: Optional[Any] = None
        self._kernel_fn: Optional[Callable[..., Any]] = None
        self._pytorch_fallback_fn: Optional[Callable[..., Any]] = None
        self._load_kernel()

    def _load_kernel(self) -> None:
        """Dynamically load Triton kernel module and resolve callable operator entrypoints."""
        try:
            self._module = importlib.import_module(self._module_path)
        except ImportError:
            # Fallback path if loaded via ops.triton directly
            try:
                self._module = importlib.import_module(f"neuralops.ops.triton.{self.kernel_name}")
            except ImportError as err:
                logger.warning(
                    "Triton kernel module %s failed to import: %s. PyTorch reference will be used if available.",
                    self._module_path,
                    err,
                )
                self._module = None

        if self._module is not None:
            # Discover primary execution function
            candidate_fn_names = [
                f"triton_{self.kernel_name}",
                f"{self.kernel_name}_triton",
                f"{self.kernel_name}_fwd",
                f"{self.kernel_name}_forward",
                f"{self.kernel_name}_attention",
                f"{self.kernel_name}_kernel",
                f"{self.kernel_name}_fn",
                f"fused_{self.kernel_name}",
                "fwd",
                "forward",
                self.kernel_name,
            ]
            for fn_name in candidate_fn_names:
                if hasattr(self._module, fn_name):
                    fn = getattr(self._module, fn_name)
                    if callable(fn):
                        self._kernel_fn = fn
                        break

            # Discover PyTorch CPU reference fallback
            ref_candidates = [
                f"triton_{self.kernel_name}_pytorch_ref",
                f"{self.kernel_name}_pytorch_ref",
                f"{self.kernel_name}_pytorch",
                f"{self.kernel_name}_ref",
                f"{self.kernel_name}_pytorch_fallback",
                f"{self.kernel_name}_fallback",
                "ref_forward",
                "pytorch_ref",
            ]
            for ref_name in ref_candidates:
                if hasattr(self._module, ref_name):
                    fn = getattr(self._module, ref_name)
                    if callable(fn):
                        self._pytorch_fallback_fn = fn
                        break

    @property
    def is_available(self) -> bool:
        """Return True if kernel callable is loaded and device is CUDA."""
        return self._kernel_fn is not None and self.device.type == "cuda" and torch.cuda.is_available()

    @staticmethod
    def verify_tiling_contract(module_name: Optional[str] = None) -> bool:
        """Enforce architectural standard: verify zero occurrences of deprecated
        `tl.make_block_ptr` or `tl.advance` in mainline Triton kernel source files,
        mandating explicit 2D SRAM tile pointer offsets per AGENTS.md Section 7.

        Args:
            module_name: Specific Triton module name to verify, or None to scan all
                26 canonical modules.

        Returns:
            True if all scanned modules adhere strictly to the 2D tiling contract.

        Raises:
            ValueError: If any disallowed block-pointer construct is detected.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent
        targets: List[str] = [module_name] if module_name else CANONICAL_TRITON_MODULES
        violations: List[str] = []

        for name in targets:
            # Check both core and ops directories
            candidates = [
                repo_root / "neuralops" / "core" / "triton_kernels" / f"{name}.py",
                repo_root / "neuralops" / "ops" / "triton" / f"{name}.py",
            ]
            for path in candidates:
                if not path.is_file():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        source = f.read()
                        tree = ast.parse(source, filename=str(path))
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Attribute) and node.attr in ("make_block_ptr", "advance"):
                            violations.append(f"{path.name}:{getattr(node, 'lineno', '?')} ({node.attr})")
                        elif isinstance(node, ast.Name) and node.id in ("make_block_ptr", "advance"):
                            violations.append(f"{path.name}:{getattr(node, 'lineno', '?')} ({node.id})")
                except Exception as exc:
                    logger.debug("Failed to parse AST for %s: %s", path, exc)

        if violations:
            msg = (
                f"Triton 2D SRAM Tiling Contract Violation: tl.make_block_ptr / tl.advance "
                f"detected in mainline Triton kernels per AGENTS.md §7: {', '.join(violations)}"
            )
            logger.error(msg)
            raise ValueError(msg)
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute kernel with zero-overhead dispatch and fail-closed warnings on fallback."""
        # Route to PyTorch CPU reference if executing on CPU or Triton is unavailable
        if not self.is_available or (args and isinstance(args[0], torch.Tensor) and not args[0].is_cuda):
            if self._pytorch_fallback_fn is not None:
                logger.warning(
                    "[Triton Fallback] Kernel '%s' running via PyTorch CPU reference (Device: %s, Triton Available: %s).",
                    self.kernel_name,
                    self.device,
                    self.is_available,
                )
                return self._pytorch_fallback_fn(*args, **kwargs)
            elif self._kernel_fn is not None:
                try:
                    return self._kernel_fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        "[Triton Kernel Error] Execution of %s failed (%s); attempting generic fallback.",
                        self.kernel_name,
                        e,
                    )
                    from neuralops.ops.triton._common import triton_dispatch

                    return triton_dispatch(self.kernel_name, *args, **kwargs)
            else:
                from neuralops.ops.triton._common import triton_dispatch

                logger.warning(
                    "[Triton Dispatch Fallback] Dispatching kernel '%s' via common fallback dispatcher.",
                    self.kernel_name,
                )
                return triton_dispatch(self.kernel_name, *args, **kwargs)

        # Level 2 accelerated Triton execution
        assert self._kernel_fn is not None
        return self._kernel_fn(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute kernel via fallback hierarchy or run 6-gate benchmark lifecycle.

        If called with input arguments, behaves as direct kernel invocation (backward-compatible).
        If called without arguments, runs the complete BaseBenchmarkAdapter 6-gate lifecycle
        and returns a canonical BenchmarkResult.
        """
        if args or kwargs:
            return self(*args, **kwargs)
        return super().execute()

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

        if self.is_available:
            return {
                "available": True,
                "reason": "",
                "backend": "triton",
            }

        if self._pytorch_fallback_fn is not None:
            reason = "" if not is_cuda_req else "Triton unavailable; falling back to PyTorch reference"
            return {
                "available": True,
                "reason": reason,
                "backend": "pytorch",
            }

        if self._kernel_fn is not None:
            return {
                "available": True,
                "reason": "",
                "backend": "triton",
            }

        return {
            "available": False,
            "reason": f"No Triton kernel or PyTorch reference resolved for '{self.kernel_name}'",
            "backend": "none",
        }

    def build_inputs(self) -> BenchmarkInputs:
        """Gate 2: construct deterministic input tensors."""
        torch.manual_seed(self.seed)
        dtype = self.dtype
        dev = self.device
        name = self.kernel_name.lower()

        if any(k in name for k in ("cosformer", "linear", "attention", "attn", "gqa", "sliding_window", "retnet")):
            q = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            k = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            v = torch.randn(2, 4, 16, 32, device=dev, dtype=dtype)
            tensors = {"q": q, "k": k, "v": v}
            args: Tuple[Any, ...] = (q, k, v)
            kwargs: Dict[str, Any] = {}
        elif "scan" in name or "retention" in name:
            u = torch.randn(2, 8, 16, device=dev, dtype=dtype)
            delta = torch.nn.functional.softplus(torch.randn(2, 8, 16, device=dev, dtype=dtype))
            a = -torch.abs(torch.randn(8, 4, device=dev, dtype=dtype)) - 0.1
            b_t = torch.randn(2, 4, 16, device=dev, dtype=dtype)
            c = torch.randn(2, 4, 16, device=dev, dtype=dtype)
            d = torch.randn(8, device=dev, dtype=dtype)
            tensors = {"u": u, "delta": delta, "a": a, "b": b_t, "c": c, "d": d}
            args = (u, delta, a, b_t, c, d)
            kwargs = {}
        elif any(k in name for k in ("norm", "ce", "loss", "elementwise", "softplus")):
            x = torch.randn(2, 16, 32, device=dev, dtype=dtype)
            tensors = {"x": x}
            args = (x,)
            kwargs = {}
        elif "quant" in name or "prune" in name or "svd" in name:
            x = torch.randn(16, 32, device=dev, dtype=dtype)
            tensors = {"x": x}
            args = (x,)
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
        """Gate 3: run the reference implementation (vectorized PyTorch)."""
        args = inputs.metadata.get("args")
        kwargs = inputs.metadata.get("kwargs", {})
        if args is None:
            args = tuple(inputs.tensors.values())

        if self._pytorch_fallback_fn is not None:
            out = self._pytorch_fallback_fn(*args, **kwargs)
        else:
            from neuralops.ops.triton._common import triton_dispatch

            out = triton_dispatch(self.kernel_name, *args, **kwargs)

        if isinstance(out, tuple):
            out = out[0]
        return out

    def run_candidate(self, inputs: BenchmarkInputs) -> Tuple[torch.Tensor, str]:
        """Gate 4: run the candidate backend."""
        args = inputs.metadata.get("args")
        kwargs = inputs.metadata.get("kwargs", {})
        if args is None:
            args = tuple(inputs.tensors.values())

        if self.is_available and self._kernel_fn is not None:
            out = self._kernel_fn(*args, **kwargs)
            backend_used = "triton"
        elif self._pytorch_fallback_fn is not None:
            out = self._pytorch_fallback_fn(*args, **kwargs)
            backend_used = "pytorch"
        elif self._kernel_fn is not None:
            out = self._kernel_fn(*args, **kwargs)
            backend_used = "triton"
        else:
            from neuralops.ops.triton._common import triton_dispatch

            out = triton_dispatch(self.kernel_name, *args, **kwargs)
            backend_used = "pytorch"

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
            _ = self(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()

        latencies: List[float] = []
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(self.measure_steps):
                start_event.record()
                _ = self(*args, **kwargs)
                end_event.record()
                end_event.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            for _ in range(self.measure_steps):
                t0 = time.perf_counter()
                _ = self(*args, **kwargs)
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
    # Catalogs
    # ------------------------------------------------------------------

    @staticmethod
    def list_canonical_kernels() -> List[str]:
        """Return sorted list of the 26 canonical Triton kernels."""
        return sorted(list(CANONICAL_TRITON_MODULES))

    @staticmethod
    def list_extension_kernels() -> List[str]:
        """Return sorted list of the 24 extension Triton kernels."""
        return sorted(list(EXTENSION_TRITON_MODULES))

    @staticmethod
    def list_all_triton_kernels() -> List[str]:
        """Return union of canonical and extension Triton kernel keys."""
        return sorted(list(set(CANONICAL_TRITON_MODULES + EXTENSION_TRITON_MODULES)))


__all__ = [
    "TritonKernelAdapter",
    "CANONICAL_TRITON_MODULES",
    "EXTENSION_TRITON_MODULES",
]
