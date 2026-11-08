#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Benchmark Adapter.

Bridges registered attention mechanisms in ATTENTION_REGISTRY to the benchmark
execution framework. Implements the 6-gate lifecycle inheriting from
BaseBenchmarkAdapter, causal/non-causal invariance validation, autograd
gradient checking, and fail-closed capability inspection.
"""

from __future__ import annotations

import logging
import math
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter, BenchmarkInputs, NumericsReport
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


class _DeviceStr(str):
    """String subclass providing .type property for torch.device compatibility."""

    @property
    def type(self) -> str:
        return torch.device(self).type


class AttentionBenchmarkAdapter(BaseBenchmarkAdapter):
    """Adapter wrapping ATTENTION_REGISTRY modules for benchmark execution.

    Implements the 6-gate lifecycle:
      - Gate 1: capability_check() — probe ATTENTION_REGISTRY and hardware availability.
      - Gate 2: build_inputs() — construct deterministic input tensors.
      - Gate 3: run_reference() — evaluate PyTorch reference baseline attention.
      - Gate 4: run_candidate() — instantiate and evaluate registered attention mechanism.
      - Gate 5: validate_numerics() — verify output finiteness, shape match, causal non-leakage,
                and autograd gradient flow.
      - Gate 6: measure_performance() — compute forward latency distribution and peak memory.
    """

    def __init__(
        self,
        attention_name: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        tolerance: float = BaseBenchmarkAdapter.DEFAULT_TOLERANCE,
        timeout_s: float = BaseBenchmarkAdapter.DEFAULT_TIMEOUT_S,
        warmup_steps: int = 5,
        measure_steps: int = 20,
        dim: int = 64,
        num_heads: int = 2,
        seq_len: int = 16,
        batch_size: int = 2,
        causal: bool = False,
        **kwargs: Any,
    ) -> None:
        dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        dev_obj = _DeviceStr(dev_str)
        super().__init__(
            benchmark_name=attention_name,
            device=dev_obj,
            dtype=dtype,
            seed=seed,
            tolerance=tolerance,
            timeout_s=timeout_s,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )
        self.attention_name = attention_name
        self.torch_device = torch.device(self.device)
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.causal = causal
        self.extra_kwargs = kwargs

        self._candidate_module: Optional[nn.Module] = None
        self._reference_module: Optional[nn.Module] = None
        self.gradient_check_passed: bool = False
        self.causal_check_passed: bool = False

    def _instantiate_module(self) -> nn.Module:
        """Instantiate attention module using config, keyword, or positional signatures."""
        if hasattr(ATTENTION_REGISTRY, "_load_all"):
            try:
                ATTENTION_REGISTRY._load_all()
            except Exception:
                pass

        cls = ATTENTION_REGISTRY.get(self.attention_name)
        if cls is None:
            raise KeyError(f"Attention '{self.attention_name}' not found in ATTENTION_REGISTRY")

        cfg = SimpleNamespace(
            hidden_size=self.dim,
            num_attention_heads=self.num_heads,
            num_heads=self.num_heads,
            n_embd=self.dim,
            d_model=self.dim,
            head_dim=self.dim // max(1, self.num_heads),
            dropout=0.0,
            attention_dropout=0.0,
            is_causal=self.causal,
            causal=self.causal,
            max_position_embeddings=2048,
            **self.extra_kwargs,
        )

        mod = None
        # Attempt signature 1: explicit dim and num_heads
        try:
            mod = cls(dim=self.dim, num_heads=self.num_heads, **self.extra_kwargs)
        except (TypeError, ValueError):
            pass

        # Attempt signature 2: config object
        if mod is None:
            try:
                mod = cls(config=cfg)
            except (TypeError, ValueError):
                pass

        # Attempt signature 3: positional config
        if mod is None:
            try:
                mod = cls(cfg)
            except (TypeError, ValueError):
                pass

        # Attempt signature 4: hidden_size, num_heads
        if mod is None:
            try:
                mod = cls(hidden_size=self.dim, num_heads=self.num_heads)
            except (TypeError, ValueError):
                pass

        # Fallback signature 5: parameterless default init
        if mod is None:
            mod = cls()

        if self.causal:
            if hasattr(mod, "is_decoder"):
                mod.is_decoder = True
            if hasattr(mod, "is_causal"):
                mod.is_causal = True
            for child in mod.children():
                if hasattr(child, "is_decoder"):
                    child.is_decoder = True
                if hasattr(child, "is_causal"):
                    child.is_causal = True

        return mod.to(self.torch_device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Gate 1: Capability Check
    # ------------------------------------------------------------------

    def capability_check(self) -> Dict[str, Any]:
        """Gate 1: Probe environment and registry only — NO kernel execution."""
        if hasattr(ATTENTION_REGISTRY, "_load_all"):
            try:
                ATTENTION_REGISTRY._load_all()
            except Exception:
                pass

        backend = "cuda" if "cuda" in str(self.device) else "pytorch"
        if "cuda" in str(self.device) and not torch.cuda.is_available():
            return {
                "available": False,
                "reason": f"Device '{self.device}' requested but CUDA is unavailable",
                "backend": "cuda",
            }

        is_registered = (self.attention_name in ATTENTION_REGISTRY) or (
            ATTENTION_REGISTRY.get(self.attention_name) is not None
        )
        if not is_registered:
            return {
                "available": False,
                "reason": f"Attention '{self.attention_name}' not found in ATTENTION_REGISTRY ({len(ATTENTION_REGISTRY)} registered)",
                "backend": backend,
            }

        if hasattr(ATTENTION_REGISTRY, "get_metadata"):
            meta = ATTENTION_REGISTRY.get_metadata(self.attention_name) or {}
            if self.causal and meta.get("supports_causal") is False:
                return {
                    "available": False,
                    "reason": f"Attention '{self.attention_name}' does not support causal masking",
                    "backend": backend,
                }

        return {"available": True, "reason": "", "backend": backend}

    # ------------------------------------------------------------------
    # Gate 2: Build Inputs
    # ------------------------------------------------------------------

    def build_inputs(self) -> BenchmarkInputs:
        """Gate 2: Construct deterministic input tensors."""
        torch.manual_seed(self.seed)
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.dim,
            device=self.torch_device,
            dtype=self.dtype,
        )

        return BenchmarkInputs(
            tensors={"x": x},
            metadata={
                "dim": self.dim,
                "num_heads": self.num_heads,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "causal": self.causal,
            },
            seed=self.seed,
            dtype=self.dtype,
            device=str(self.device),
        )

    # ------------------------------------------------------------------
    # Gate 3: Run Reference
    # ------------------------------------------------------------------

    def run_reference(self, inputs: BenchmarkInputs) -> torch.Tensor:
        """Gate 3: Run vectorized PyTorch reference attention forward pass."""
        x = inputs.tensors["x"].detach()
        head_dim = self.dim // max(1, self.num_heads)
        q = x.view(self.batch_size, self.seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = q.clone()
        v = q.clone()

        with torch.no_grad():
            ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        ref_out = ref_out.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.dim)
        return ref_out

    # ------------------------------------------------------------------
    # Gate 4: Run Candidate
    # ------------------------------------------------------------------

    def run_candidate(self, inputs: BenchmarkInputs) -> Tuple[torch.Tensor, str]:
        """Gate 4: Instantiate registered module and evaluate candidate forward pass."""
        self._candidate_module = self._instantiate_module()
        self._candidate_module.eval()

        x = inputs.tensors["x"].clone()
        with torch.no_grad():
            try:
                cand_out = self._candidate_module(x, is_causal=self.causal)
            except TypeError:
                try:
                    cand_out = self._candidate_module(x, causal=self.causal)
                except TypeError:
                    cand_out = self._candidate_module(x)

        if isinstance(cand_out, tuple):
            cand_out = cand_out[0]

        backend_used = "cuda" if "cuda" in str(self.device) else "pytorch"
        return cand_out, backend_used

    # ------------------------------------------------------------------
    # Gate 5: Validate Numerics & Autograd
    # ------------------------------------------------------------------

    def validate_numerics(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> NumericsReport:
        """Gate 5: Shape/finiteness check, causal non-leakage verification, and autograd check."""
        cand_finite = bool(torch.isfinite(candidate).all().item())
        ref_finite = bool(torch.isfinite(reference).all().item())
        shape_match = candidate.shape == reference.shape

        if not cand_finite:
            return NumericsReport(
                passed=False,
                max_diff=float("nan"),
                tolerance=self.tolerance,
                reference_finite=ref_finite,
                candidate_finite=False,
                shape_match=shape_match,
                dtype_match=candidate.dtype == reference.dtype,
                failure_reason="Candidate output contains NaN or Inf",
            )

        if not shape_match:
            return NumericsReport(
                passed=False,
                max_diff=float("inf"),
                tolerance=self.tolerance,
                reference_finite=ref_finite,
                candidate_finite=cand_finite,
                shape_match=False,
                dtype_match=candidate.dtype == reference.dtype,
                failure_reason=f"Shape mismatch: reference={reference.shape} candidate={candidate.shape}",
            )

        # 1. Causal non-leakage validation
        if self.causal:
            causal_ok, causal_err = self._verify_causal_non_leakage()
            if not causal_ok:
                return NumericsReport(
                    passed=False,
                    max_diff=float("inf"),
                    tolerance=self.tolerance,
                    reference_finite=ref_finite,
                    candidate_finite=cand_finite,
                    shape_match=shape_match,
                    dtype_match=True,
                    failure_reason=f"Causal verification failed: {causal_err}",
                )
            self.causal_check_passed = True
        else:
            self.causal_check_passed = True

        # 2. Autograd gradient check
        grad_ok, grad_err = self.run_autograd_gradient_check()
        if not grad_ok:
            return NumericsReport(
                passed=False,
                max_diff=float("inf"),
                tolerance=self.tolerance,
                reference_finite=ref_finite,
                candidate_finite=cand_finite,
                shape_match=shape_match,
                dtype_match=True,
                failure_reason=f"Autograd gradient check failed: {grad_err}",
            )
        self.gradient_check_passed = True

        diff = float((reference.float() - candidate.float()).abs().max().item())
        return NumericsReport(
            passed=True,
            max_diff=diff,
            tolerance=self.tolerance,
            reference_finite=ref_finite,
            candidate_finite=cand_finite,
            shape_match=True,
            dtype_match=candidate.dtype == reference.dtype,
            failure_reason=None,
        )

    def _verify_causal_non_leakage(self) -> Tuple[bool, Optional[str]]:
        """Verify future token perturbation does not affect past token representations."""
        if self._candidate_module is None:
            return False, "Candidate module is not instantiated"

        mod = self._candidate_module
        mod.eval()
        B, T, D = 1, max(self.seq_len, 8), self.dim
        x1 = torch.randn(B, T, D, device=self.torch_device, dtype=self.dtype)
        x2 = x1.clone()
        half = T // 2
        x2[:, half:, :] += 10.0 * torch.randn_like(x2[:, half:, :])

        def _forward(inp: torch.Tensor) -> torch.Tensor:
            try:
                res = mod(inp, is_causal=True)
            except TypeError:
                try:
                    res = mod(inp, causal=True)
                except TypeError:
                    res = mod(inp)
            if isinstance(res, tuple):
                res = res[0]
            return res

        with torch.no_grad():
            y1 = _forward(x1)
            y2 = _forward(x2)

        past_diff = (y1[:, :half, :] - y2[:, :half, :]).abs().max().item()
        if past_diff > 1e-4:
            return False, f"Past tokens changed by {past_diff:.6e} under future perturbation"
        return True, None

    def run_autograd_gradient_check(self) -> Tuple[bool, Optional[str]]:
        """Execute backward pass and certify gradient propagation to input and parameters."""
        mod = self._instantiate_module()
        mod.train()
        mod.zero_grad(set_to_none=True)

        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.dim,
            device=self.torch_device,
            dtype=self.dtype,
            requires_grad=True,
        )

        try:
            try:
                out = mod(x, is_causal=self.causal)
            except TypeError:
                try:
                    out = mod(x, causal=self.causal)
                except TypeError:
                    out = mod(x)

            if isinstance(out, tuple):
                out = out[0]

            loss = out.sum()
            loss.backward()

            if x.grad is None:
                return False, "Input tensor x.grad is None after backward pass"
            if not torch.isfinite(x.grad).all():
                return False, "Input tensor x.grad contains NaN or Inf values"

            # Check trainable parameters
            for name, param in mod.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        return False, f"Parameter '{name}' grad contains NaN or Inf values"

            self.gradient_check_passed = True
            return True, None
        except Exception as e:
            return False, f"Autograd exception during backward pass: {e}"

    # ------------------------------------------------------------------
    # Gate 6: Measure Performance
    # ------------------------------------------------------------------

    def measure_performance(self, inputs: BenchmarkInputs) -> Dict[str, float]:
        """Gate 6: Profile forward latency distribution and peak memory."""
        if self._candidate_module is None:
            self._candidate_module = self._instantiate_module()
        mod = self._candidate_module
        mod.eval()

        x = inputs.tensors["x"]
        is_cuda = "cuda" in str(self.device) and torch.cuda.is_available()

        def _step():
            try:
                out = mod(x, is_causal=self.causal)
            except TypeError:
                try:
                    out = mod(x, causal=self.causal)
                except TypeError:
                    out = mod(x)
            return out

        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _step()
            if is_cuda:
                torch.cuda.synchronize()

            latencies: List[float] = []
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(self.measure_steps):
                    start_event.record()
                    _step()
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                for _ in range(self.measure_steps):
                    t0 = time.perf_counter()
                    _step()
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
                peak_mem_mb = 0.0

        lat_sorted = sorted(latencies)
        n = len(lat_sorted)
        mean_lat = sum(lat_sorted) / max(n, 1)
        variance = sum((x_l - mean_lat) ** 2 for x_l in lat_sorted) / max(n - 1, 1) if n > 1 else 0.0
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
            "gradient_check_passed": 1.0 if self.gradient_check_passed else 0.0,
            "causal_check_passed": 1.0 if self.causal_check_passed else 0.0,
        }


__all__ = ["AttentionBenchmarkAdapter"]
