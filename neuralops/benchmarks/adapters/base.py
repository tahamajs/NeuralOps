#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
BaseBenchmarkAdapter — abstract 3-gate lifecycle for all NeuralOps benchmark adapters.

Gate order:
  1. capability_check()   — environment probe, NO kernel execution
  2. build_inputs()       — deterministic tensor construction, seed recorded
  3. run_reference()      — reference backend (always PyTorch CPU/fp32 unless overridden)
  4. run_candidate()      — backend under test; records actual backend_used
  5. validate_numerics()  — shape, dtype, finite, max_diff vs tolerance
  6. measure_performance()— timing ONLY after Gate 5 passes

Rules (from AGENTS.md §3 / spec §3):
  - Device mismatch must raise RuntimeError before any kernel runs.
  - CUDA unavailable with --device cuda → RuntimeError, exit code 3.
  - NaN/Inf in candidate output → FAIL_NUMERICS, never SUCCESS.
  - max_diff > tolerance → FAIL_NUMERICS.
  - Any unhandled exception is stored in result, not re-raised to caller.
  - backend_used must reflect the actual executed backend.
"""

from __future__ import annotations

import abc
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus

# ---------------------------------------------------------------------------
# Input descriptor
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkInputs:
    """Deterministic input tensors for one benchmark run."""

    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class NumericsReport:
    """Outcome of the validate_numerics gate."""

    passed: bool
    max_diff: float
    tolerance: float
    reference_finite: bool
    candidate_finite: bool
    shape_match: bool
    dtype_match: bool
    failure_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseBenchmarkAdapter(abc.ABC):
    """Abstract base for all NeuralOps benchmark adapters.

    Concrete subclasses must implement the six abstract methods.
    Call ``execute()`` to run the full gated lifecycle.
    """

    #: Default numerical tolerance for fp32.  Subclasses may override.
    DEFAULT_TOLERANCE: float = 1e-4
    #: Default maximum allowed benchmark duration in seconds.
    DEFAULT_TIMEOUT_S: float = 120.0

    def __init__(
        self,
        benchmark_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        tolerance: float = DEFAULT_TOLERANCE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        warmup_steps: int = 5,
        measure_steps: int = 20,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.tolerance = tolerance
        self.timeout_s = timeout_s
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps

        # Fail closed: validate device before any work.
        self._check_device_availability()

    # ------------------------------------------------------------------
    # Device validation (fail-closed per AGENTS.md §3)
    # ------------------------------------------------------------------

    def _check_device_availability(self) -> None:
        """Raise RuntimeError if the requested device is unavailable."""
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"[{self.benchmark_name}] Device 'cuda' was requested but CUDA is "
                    "unavailable on this host. Set --device cpu or run on a CUDA host."
                )
        elif self.device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"[{self.benchmark_name}] Device '{self.device}' was requested but "
                    "CUDA is unavailable on this host."
                )
            idx = int(self.device.split(":")[1])
            if idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"[{self.benchmark_name}] Device '{self.device}' requested but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are present."
                )

    # ------------------------------------------------------------------
    # Abstract gate methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def capability_check(self) -> Dict[str, Any]:
        """Gate 1: probe environment only — NO kernel execution.

        Returns a dict with at minimum:
          - "available": bool
          - "reason": str (empty if available)
          - "backend": str
        """

    @abc.abstractmethod
    def build_inputs(self) -> BenchmarkInputs:
        """Gate 2: construct deterministic input tensors.

        Must record ``seed`` and ``dtype`` in the returned object.
        """

    @abc.abstractmethod
    def run_reference(self, inputs: BenchmarkInputs) -> torch.Tensor:
        """Gate 3: run the reference implementation (usually vectorized PyTorch).

        Must NOT use the native CUDA or Triton backend.
        """

    @abc.abstractmethod
    def run_candidate(self, inputs: BenchmarkInputs) -> Tuple[torch.Tensor, str]:
        """Gate 4: run the candidate backend.

        Returns ``(output_tensor, actual_backend_used)`` where
        ``actual_backend_used`` is one of ``"cuda"``, ``"triton"``, ``"pytorch"``.
        Never return a backend label that wasn't actually executed.
        """

    @abc.abstractmethod
    def validate_numerics(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> NumericsReport:
        """Gate 5: compare reference vs candidate numerically.

        Base implementation provided via ``_default_validate_numerics``;
        subclasses may call it or override entirely.
        """

    @abc.abstractmethod
    def measure_performance(self, inputs: BenchmarkInputs) -> Dict[str, float]:
        """Gate 6: timing ONLY after Gate 5 passed.

        Returns metrics dict with keys such as:
          - "latency_ms_mean", "latency_ms_std"
          - "throughput_tokens_per_sec"
          - "memory_mb_peak"
        Never hard-code these values.
        """

    # ------------------------------------------------------------------
    # Default validation helper (callable from subclasses)
    # ------------------------------------------------------------------

    def _default_validate_numerics(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        tolerance: Optional[float] = None,
    ) -> NumericsReport:
        """Standard shape/dtype/finite/max_diff validation."""
        tol = tolerance if tolerance is not None else self.tolerance

        shape_match = reference.shape == candidate.shape
        dtype_match = reference.dtype == candidate.dtype

        ref_flat = reference.detach().float()
        cand_flat = candidate.detach().float()

        ref_finite = bool(torch.isfinite(ref_flat).all())
        cand_finite = bool(torch.isfinite(cand_flat).all())

        if not shape_match:
            return NumericsReport(
                passed=False,
                max_diff=float("inf"),
                tolerance=tol,
                reference_finite=ref_finite,
                candidate_finite=cand_finite,
                shape_match=False,
                dtype_match=dtype_match,
                failure_reason=(f"Shape mismatch: reference={reference.shape} " f"candidate={candidate.shape}"),
            )

        if not cand_finite:
            return NumericsReport(
                passed=False,
                max_diff=float("nan"),
                tolerance=tol,
                reference_finite=ref_finite,
                candidate_finite=False,
                shape_match=shape_match,
                dtype_match=dtype_match,
                failure_reason="Candidate output contains NaN or Inf",
            )

        if not ref_finite:
            return NumericsReport(
                passed=False,
                max_diff=float("nan"),
                tolerance=tol,
                reference_finite=False,
                candidate_finite=cand_finite,
                shape_match=shape_match,
                dtype_match=dtype_match,
                failure_reason="Reference output contains NaN or Inf",
            )

        max_diff = float((ref_flat - cand_flat).abs().max().item())
        passed = (max_diff <= tol) and shape_match and dtype_match and cand_finite

        reason: Optional[str] = None
        if not passed:
            if max_diff > tol:
                reason = f"max_diff={max_diff:.6e} exceeds tolerance={tol:.6e}"
            elif not dtype_match:
                reason = f"dtype mismatch: reference={reference.dtype} " f"candidate={candidate.dtype}"

        return NumericsReport(
            passed=passed,
            max_diff=max_diff,
            tolerance=tol,
            reference_finite=ref_finite,
            candidate_finite=cand_finite,
            shape_match=shape_match,
            dtype_match=dtype_match,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Orchestrated execute()
    # ------------------------------------------------------------------

    def execute(self) -> BenchmarkResult:
        """Run the full 6-gate lifecycle and return a BenchmarkResult.

        Exceptions in any gate are captured into the result — they do NOT
        propagate to the caller (isolation contract).  The status will be
        FAILED or DEPENDENCY_ERROR as appropriate.
        """
        import uuid as _uuid

        run_id = str(_uuid.uuid4())
        backend_used: str = "unknown"
        # Store adapter-specific metadata in the 'metadata' dict field.
        meta: Dict[str, Any] = {
            "benchmark_name": self.benchmark_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "seed": self.seed,
            "tolerance": self.tolerance,
        }
        # 'module' in BenchmarkResult is a dict with module identity fields.
        module_info: Dict[str, Any] = {"name": self.benchmark_name}

        def _make_result(status: str, error: Optional[str] = None) -> BenchmarkResult:
            return BenchmarkResult(
                run_id=run_id,
                module=module_info,
                status=status,
                error=error,
                metadata=dict(meta),
            )

        # Gate 1: capability check
        try:
            cap = self.capability_check()
            if not cap.get("available", True):
                return _make_result(
                    str(BenchmarkStatus.DEPENDENCY_ERROR.value),
                    cap.get("reason", "capability check failed"),
                )
            backend_used = cap.get("backend", "unknown")
            meta["backend_used"] = backend_used
        except Exception:
            tb = traceback.format_exc()
            return _make_result(str(BenchmarkStatus.DEPENDENCY_ERROR.value), tb)

        # Gate 2: build inputs
        try:
            inputs = self.build_inputs()
        except Exception:
            tb = traceback.format_exc()
            return _make_result(str(BenchmarkStatus.FAILED.value), tb)

        # Gate 3: reference
        try:
            reference_out = self.run_reference(inputs)
        except Exception:
            tb = traceback.format_exc()
            return _make_result(
                str(BenchmarkStatus.FAILED.value),
                f"Reference run failed:\n{tb}",
            )

        # Gate 4: candidate
        try:
            candidate_out, backend_used = self.run_candidate(inputs)
            meta["backend_used"] = backend_used
        except Exception:
            tb = traceback.format_exc()
            return _make_result(
                str(BenchmarkStatus.FAILED.value),
                f"Candidate run failed (backend={backend_used}):\n{tb}",
            )

        # Gate 5: validate numerics
        try:
            report = self.validate_numerics(reference_out, candidate_out)
            meta["max_diff"] = report.max_diff
            meta["tolerance"] = report.tolerance
            meta["numerics_passed"] = report.passed

            if not report.passed:
                return _make_result(
                    str(BenchmarkStatus.FAIL_NUMERICS.value),
                    report.failure_reason or "Numerics gate failed",
                )
        except Exception:
            tb = traceback.format_exc()
            return _make_result(
                str(BenchmarkStatus.FAILED.value),
                f"validate_numerics raised:\n{tb}",
            )

        # Gate 6: measure performance
        try:
            perf = self.measure_performance(inputs)
            meta.update(perf)
        except Exception:
            tb = traceback.format_exc()
            # Correctness passed — record partial failure with error note.
            return _make_result(
                str(BenchmarkStatus.FAILED.value),
                f"measure_performance raised after numerics passed:\n{tb}",
            )

        return _make_result(str(BenchmarkStatus.SUCCESS.value))
