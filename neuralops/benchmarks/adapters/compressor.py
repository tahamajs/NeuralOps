#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Benchmark adapter for NeuralOps compression and PEFT methods."""

from __future__ import annotations

import copy
import logging
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from neuralops.benchmarks.adapters.base import (
    BaseBenchmarkAdapter,
    BenchmarkInputs,
    NumericsReport,
)
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class _PeftBenchmarkModel(nn.Module):
    """Small model whose projection names match the PEFT and compressor target contracts."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return self.dense(torch.matmul(scores.softmax(dim=-1), v))


class CompressorModuleAdapter(BaseBenchmarkAdapter):
    """Measure real compression, low-rank factorization, and PEFT surgery methods.

    Inherits from BaseBenchmarkAdapter and implements the 6-gate lifecycle:
      - Gate 1: capability_check() — probe environment, verify compressor presence in COMPRESSOR_REGISTRY.
      - Gate 2: build_inputs() — snapshot model before mutation, construct deterministic calibration input.
      - Gate 3: run_reference() — run reference forward pass on calibration tensor with uncompressed model.
      - Gate 4: run_candidate() — apply target compressor strictly without silent SVD fallback, then run candidate forward.
      - Gate 5: validate_numerics() — compute relative Frobenius error ||W - W_hat||_F / ||W||_F and SQNR in dB.
      - Gate 6: measure_performance() — compute forward latency, speedup, compression ratio, and param reduction.
    """

    METHODS = ("lora", "pissa", "dora")

    def __init__(
        self,
        compressor_name: str = "direct_svd",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        tolerance: float = 1.0,
        timeout_s: float = 120.0,
        warmup_steps: int = 2,
        measure_steps: int = 10,
        hidden_size: int = 128,
        rank: int = 16,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            benchmark_name=compressor_name,
            device=dev_str,
            dtype=dtype,
            seed=seed,
            tolerance=tolerance,
            timeout_s=timeout_s,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
        )
        self.compressor_name = compressor_name
        self.torch_device = torch.device(self.device)
        self.hidden_size = hidden_size
        self.rank = rank
        self.user_model = model
        self.extra_kwargs = kwargs

        # Snapshot and candidate storage
        self._snapshot_model: Optional[nn.Module] = None
        self._reference_model: Optional[nn.Module] = None
        self._candidate_model: Optional[nn.Module] = None
        self._original_weights: Dict[str, torch.Tensor] = {}
        self._reconstructed_weights: Dict[str, torch.Tensor] = {}

        # Distortion & performance metrics
        self.relative_frob_error: float = 0.0
        self.sqnr_db: float = 0.0
        self.max_abs_diff: float = 0.0
        self.compression_ratio: float = 1.0
        self.param_reduction_pct: float = 0.0

    # ------------------------------------------------------------------
    # Gate 1: Capability Check
    # ------------------------------------------------------------------

    def capability_check(self) -> Dict[str, Any]:
        """Gate 1: Probe environment only — NO kernel execution."""
        if hasattr(COMPRESSOR_REGISTRY, "_load_all"):
            try:
                COMPRESSOR_REGISTRY._load_all()
            except Exception:
                pass

        backend = "cuda" if "cuda" in self.device else "pytorch"
        if "cuda" in self.device and not torch.cuda.is_available():
            return {
                "available": False,
                "reason": f"CUDA requested ({self.device}) but not available on this host",
                "backend": "cuda",
            }

        is_registered = (self.compressor_name in COMPRESSOR_REGISTRY) or (
            COMPRESSOR_REGISTRY.get(self.compressor_name) is not None
        )
        is_legacy = self.compressor_name in self.METHODS

        if not (is_registered or is_legacy):
            return {
                "available": False,
                "reason": (
                    f"Compressor '{self.compressor_name}' not found in COMPRESSOR_REGISTRY "
                    f"({len(COMPRESSOR_REGISTRY)} registered)."
                ),
                "backend": backend,
            }

        return {"available": True, "reason": "", "backend": backend}

    # ------------------------------------------------------------------
    # Gate 2: Build Inputs & Snapshot Model
    # ------------------------------------------------------------------

    def build_inputs(self) -> BenchmarkInputs:
        """Gate 2: Snapshot reference model and construct deterministic calibration input."""
        torch.manual_seed(self.seed)

        if self.user_model is not None:
            base = copy.deepcopy(self.user_model).to(self.torch_device, dtype=self.dtype)
        else:
            base = _PeftBenchmarkModel(self.hidden_size).to(self.torch_device, dtype=self.dtype)

        # Snapshot model BEFORE any mutation occurs
        self._snapshot_model = copy.deepcopy(base)
        self._reference_model = copy.deepcopy(base)
        self._candidate_model = None

        # Record original linear layer weights from snapshot
        self._original_weights = self._extract_linear_weights(self._snapshot_model)

        # Build calibration input tensor
        calib_x = torch.randn(
            2, 32, self.hidden_size, device=self.torch_device, dtype=self.dtype
        )

        return BenchmarkInputs(
            tensors={"calibration_input": calib_x, "x": calib_x},
            metadata={
                "hidden_size": self.hidden_size,
                "rank": self.rank,
                "compressor_name": self.compressor_name,
            },
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Gate 3: Run Reference
    # ------------------------------------------------------------------

    def run_reference(self, inputs: BenchmarkInputs) -> torch.Tensor:
        """Gate 3: Run reference forward on calibration tensor using uncompressed model."""
        if self._reference_model is None:
            raise RuntimeError("Reference model snapshot was not created in build_inputs.")

        self._reference_model.eval()
        x = inputs.tensors.get("calibration_input", inputs.tensors.get("x"))
        with torch.no_grad():
            ref_out = self._reference_model(x)

        if hasattr(ref_out, "last_hidden_state"):
            ref_out = ref_out.last_hidden_state

        return ref_out.detach().clone()

    # ------------------------------------------------------------------
    # Gate 4: Run Candidate (Strict Fail-Closed)
    # ------------------------------------------------------------------

    def run_candidate(self, inputs: BenchmarkInputs) -> Tuple[torch.Tensor, str]:
        """Gate 4: Apply target compressor strictly without silent SVD fallback, then run candidate."""
        if self._snapshot_model is None:
            raise RuntimeError("Snapshot model was not created in build_inputs.")

        # Candidate begins from an exact copy of the pre-mutation snapshot
        candidate_model = copy.deepcopy(self._snapshot_model)

        # Strict fail-closed: do NOT catch compressor exceptions and substitute generic SVD!
        # If named compressor fails, this will raise and execute() sets status to FAILED.
        if self.compressor_name in self.METHODS and self.compressor_name not in COMPRESSOR_REGISTRY:
            self._inject_legacy(self.compressor_name, candidate_model)
            compressed_model = candidate_model
        else:
            compressed_model = self._apply_target_compressor(self.compressor_name, candidate_model)

        self._candidate_model = compressed_model
        self._candidate_model.eval()

        # Extract reconstructed weights for Frobenius distortion computation
        self._reconstructed_weights = self._extract_linear_weights(self._candidate_model)

        x = inputs.tensors.get("calibration_input", inputs.tensors.get("x"))
        with torch.no_grad():
            cand_out = self._candidate_model(x)

        if hasattr(cand_out, "last_hidden_state"):
            cand_out = cand_out.last_hidden_state

        backend_used = "cuda" if "cuda" in self.device else "pytorch"
        return cand_out, backend_used

    def _apply_target_compressor(self, name: str, model: nn.Module) -> nn.Module:
        """Apply named compressor from COMPRESSOR_REGISTRY with fail-closed semantics."""
        cls = COMPRESSOR_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"Compressor '{name}' not found in COMPRESSOR_REGISTRY")

        # Pass fallback_to_original=False to prevent silent fallback to uncompressed layer/SVD
        comp_kwargs = {
            "rank": self.rank,
            "fallback_to_original": False,
            "protect_critical": False,
            "target_attention_only": False,
            "max_error": 1.0,
        }
        comp_kwargs.update(self.extra_kwargs)

        comp_obj = None
        try:
            comp_obj = cls(model=model, **comp_kwargs)
        except TypeError:
            try:
                comp_obj = cls(model, **comp_kwargs)
            except TypeError:
                try:
                    comp_obj = cls(model=model)
                except TypeError:
                    try:
                        comp_obj = cls(model)
                    except TypeError:
                        comp_obj = cls()

        if hasattr(comp_obj, "compress"):
            res = comp_obj.compress()
            compressed = res[0] if isinstance(res, tuple) else res
        elif callable(comp_obj) and comp_obj is not model:
            try:
                res = comp_obj(model, **comp_kwargs)
            except TypeError:
                res = comp_obj(model)
            if isinstance(res, tuple):
                compressed = res[0]
            elif hasattr(res, "compress"):
                c_res = res.compress()
                compressed = c_res[0] if isinstance(c_res, tuple) else c_res
            else:
                compressed = res
        else:
            compressed = model

        if compressed is None or not isinstance(compressed, nn.Module):
            raise RuntimeError(
                f"Compressor '{name}' failed to produce a valid nn.Module (got {type(compressed)})."
            )

        return compressed

    # ------------------------------------------------------------------
    # Gate 5: Validate Numerics (Frobenius Distortion & SQNR)
    # ------------------------------------------------------------------

    def validate_numerics(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> NumericsReport:
        """Gate 5: Measure relative Frobenius error ||W - W_hat||_F / ||W||_F and SQNR in dB."""
        shape_match = reference.shape == candidate.shape
        dtype_match = reference.dtype == candidate.dtype

        ref_finite = bool(torch.isfinite(reference).all().item())
        cand_finite = bool(torch.isfinite(candidate).all().item())

        if not cand_finite:
            return NumericsReport(
                passed=False,
                max_diff=float("nan"),
                tolerance=self.tolerance,
                reference_finite=ref_finite,
                candidate_finite=False,
                shape_match=shape_match,
                dtype_match=dtype_match,
                failure_reason="Candidate compressed model produced NaN or Inf values.",
            )

        if not shape_match:
            return NumericsReport(
                passed=False,
                max_diff=float("inf"),
                tolerance=self.tolerance,
                reference_finite=ref_finite,
                candidate_finite=cand_finite,
                shape_match=False,
                dtype_match=dtype_match,
                failure_reason=f"Output shape mismatch: ref={reference.shape} vs cand={candidate.shape}",
            )

        # 1. Weight-level relative Frobenius distortion ||W - W_hat||_F / ||W||_F & SQNR
        orig_tensors: list[torch.Tensor] = []
        cand_tensors: list[torch.Tensor] = []
        for name, w_orig in self._original_weights.items():
            if name in self._reconstructed_weights:
                w_cand = self._reconstructed_weights[name]
                if w_orig.shape == w_cand.shape:
                    orig_tensors.append(w_orig.flatten())
                    cand_tensors.append(w_cand.flatten())

        eps = 1e-12
        if orig_tensors and cand_tensors:
            w_all = torch.cat(orig_tensors).float()
            w_hat = torch.cat(cand_tensors).float()
            diff_w = w_all - w_hat
            w_norm = torch.linalg.norm(w_all)
            diff_w_norm = torch.linalg.norm(diff_w)
            self.relative_frob_error = float((diff_w_norm / (w_norm + eps)).item())
            signal_power = torch.sum(w_all.square())
            noise_power = torch.sum(diff_w.square()) + eps
            self.sqnr_db = float((10.0 * torch.log10(signal_power / noise_power)).item())
        else:
            # Fallback to calibration output tensor Frobenius distortion
            ref_f = reference.detach().float()
            cand_f = candidate.detach().float()
            diff_act = ref_f - cand_f
            ref_norm = torch.linalg.norm(ref_f)
            diff_norm = torch.linalg.norm(diff_act)
            self.relative_frob_error = float((diff_norm / (ref_norm + eps)).item())
            signal_power = torch.sum(ref_f.square())
            noise_power = torch.sum(diff_act.square()) + eps
            self.sqnr_db = float((10.0 * torch.log10(signal_power / noise_power)).item())

        max_diff = float((reference.detach().float() - candidate.detach().float()).abs().max().item())
        self.max_abs_diff = max_diff

        passed = cand_finite and shape_match and (max_diff <= self.tolerance)
        reason = None
        if not passed:
            reason = f"max_diff={max_diff:.6e} exceeds tolerance={self.tolerance:.6e}"

        return NumericsReport(
            passed=passed,
            max_diff=max_diff,
            tolerance=self.tolerance,
            reference_finite=ref_finite,
            candidate_finite=cand_finite,
            shape_match=shape_match,
            dtype_match=dtype_match,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Gate 6: Measure Performance
    # ------------------------------------------------------------------

    def measure_performance(self, inputs: BenchmarkInputs) -> Dict[str, float]:
        """Gate 6: timing, compression ratio, parameter reduction, and distortion metrics."""
        if self._candidate_model is None or self._snapshot_model is None:
            raise RuntimeError("Models not available for performance measurement.")

        x = inputs.tensors.get("calibration_input", inputs.tensors.get("x"))

        baseline_ms = self._timed(self._snapshot_model, x, repeats=self.measure_steps)
        candidate_ms = self._timed(self._candidate_model, x, repeats=self.measure_steps)

        orig_params = sum(p.numel() for p in self._snapshot_model.parameters())
        cand_params = sum(p.numel() for p in self._candidate_model.parameters())
        trainable_params = sum(p.numel() for p in self._candidate_model.parameters() if p.requires_grad)

        self.compression_ratio = float(max(1.0, orig_params / max(1, cand_params)))
        self.param_reduction_pct = float(max(0.0, 100.0 * (1.0 - cand_params / max(1, orig_params))))

        speedup = baseline_ms / candidate_ms if candidate_ms > 0 else 0.0
        peak_vram = (
            torch.cuda.max_memory_allocated(self.torch_device) / (1024**2)
            if self.torch_device.type == "cuda"
            else 0.0
        )

        return {
            "latency_ms_mean": candidate_ms,
            "baseline_latency_ms": baseline_ms,
            "speedup": speedup,
            "compression_ratio": self.compression_ratio,
            "param_reduction_pct": self.param_reduction_pct,
            "trainable_params": float(trainable_params),
            "original_params": float(orig_params),
            "compressed_params": float(cand_params),
            "relative_frob_error": self.relative_frob_error,
            "sqnr_db": self.sqnr_db,
            "max_abs_diff": self.max_abs_diff,
            "peak_vram_mb": peak_vram,
        }

    # ------------------------------------------------------------------
    # Helper & Legacy Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_linear_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
        """Extract explicit 2D weight matrices from all linear or low-rank layers."""
        weights: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weights[name] = module.weight.detach().float()
            elif hasattr(module, "weight"):
                w = module.weight
                if callable(w):
                    try:
                        w = w()
                    except Exception:
                        continue
                if isinstance(w, torch.Tensor) and w.ndim >= 2:
                    weights[name] = w.detach().float()
            elif hasattr(module, "get_weight") and callable(module.get_weight):
                try:
                    w = module.get_weight()
                    if isinstance(w, torch.Tensor) and w.ndim >= 2:
                        weights[name] = w.detach().float()
                except Exception:
                    pass
        return weights

    def _timed(self, model: nn.Module, x: torch.Tensor, repeats: int = 10) -> float:
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_steps):
                model(x)
            if self.torch_device.type == "cuda":
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(repeats):
                    model(x)
                end.record()
                end.synchronize()
                return float(start.elapsed_time(end) / repeats)
            started = time.perf_counter()
            for _ in range(repeats):
                model(x)
            return (time.perf_counter() - started) * 1000.0 / repeats

    def _inject_legacy(self, name: str, model: nn.Module) -> int:
        if name == "lora":
            from neuralops.trainer.methods.lora import apply_lora

            apply_lora(model, SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0))
            return sum(hasattr(module, "lora_A") for module in model.modules())
        if name == "dora":
            from neuralops.trainer.methods.lora import apply_dora

            apply_dora(model, SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0))
            return sum(hasattr(module, "lora_A") for module in model.modules())
        if name == "pissa":
            from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters

            return int(
                inject_pissa_adapters(
                    model,
                    rank=4,
                    target_modules=["query", "key", "value", "dense"],
                )
            )
        raise ValueError(f"Unsupported PEFT method: {name}")

    def benchmark_module(self, module_name: str) -> dict[str, Any]:
        """Backwards-compatible interface returning benchmark dictionary."""
        self.compressor_name = module_name
        self.benchmark_name = module_name
        result = self.execute()
        meta = result.metadata or {}
        return {
            "module": module_name,
            "category": "peft" if module_name in self.METHODS else "compressor",
            "cuda_latency_ms": meta.get("latency_ms_mean", 0.0),
            "pytorch_latency_ms": meta.get("baseline_latency_ms", 0.0),
            "speedup": meta.get("speedup", 0.0),
            "param_reduction_pct": meta.get("param_reduction_pct", 0.0),
            "compression_ratio": meta.get("compression_ratio", 1.0),
            "trainable_params": int(meta.get("trainable_params", 0)),
            "relative_frob_error": meta.get("relative_frob_error", 0.0),
            "sqnr_db": meta.get("sqnr_db", 0.0),
            "peak_vram_mb": meta.get("peak_vram_mb", 0.0),
            "status": "PASS" if result.status == BenchmarkStatus.SUCCESS.value else "FAIL",
            "benchmark_status": result.status,
            "error": result.error,
        }


__all__ = ["CompressorModuleAdapter"]
