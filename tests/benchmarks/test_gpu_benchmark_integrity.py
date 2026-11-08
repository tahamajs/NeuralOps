#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for fail-closed GPU benchmark device selection."""

import pytest
import torch
from torch import nn

from neuralops.benchmarks.benchmark_all_26_comprehensive import benchmark_single_kernel
from neuralops.benchmarks.benchmark_all_26_kernels import run_all_26_kernels
from neuralops.benchmarks.quantization import run_benchmark as run_quantization_benchmark
from neuralops.benchmarks.moe import run_benchmark as run_moe_benchmark
from neuralops.benchmarks.rope import run_benchmark as run_rope_benchmark
from neuralops.benchmarks.sampling import run_benchmark as run_sampling_benchmark
from neuralops.benchmarks.serving import run_benchmark as run_serving_benchmark
from neuralops.benchmarks.ssm import run_benchmark as run_ssm_benchmark
from neuralops.benchmarks.verify_ops import _make_inputs
from neuralops.benchmarks.legacy_scripts.gpu_master_suite import _resolve_requested_device
from neuralops.benchmarks.legacy_scripts.gpu_master_suite import benchmark_single_attention, benchmark_single_compressor
from neuralops.benchmarks.adapters.compressor import CompressorModuleAdapter
from neuralops.benchmarks.adapters.base import BaseBenchmarkAdapter
from neuralops.benchmarks.results.schema import BenchmarkResult, BenchmarkStatus
from neuralops.core.registry import COMPRESSOR_REGISTRY


def test_auto_device_resolves_to_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert _resolve_requested_device("auto") == "cuda"


def test_cuda_request_does_not_fallback_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _resolve_requested_device("cuda:0")


def test_direct_kernel_marks_numerical_mismatch_as_failure():
    value = torch.ones(4)
    result = benchmark_single_kernel(
        "mismatch",
        "test",
        "4",
        lambda tensor: tensor,
        lambda tensor: tensor + 1.0,
        value,
        warmup=0,
        reps=1,
        device=torch.device("cpu"),
    )
    assert result["status"] == "FAIL_NUMERICS"


def test_26_kernel_runner_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="GPU-only"):
        run_all_26_kernels(device_str="cpu")


@pytest.mark.parametrize(
    "benchmark",
    [
        run_quantization_benchmark,
        run_moe_benchmark,
        run_rope_benchmark,
        run_sampling_benchmark,
        run_serving_benchmark,
        run_ssm_benchmark,
    ],
)
def test_simple_benchmarks_reject_unavailable_cuda(monkeypatch, benchmark):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        benchmark(device="cuda")


def test_operation_input_factory_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    class Config:
        input_factory = lambda specs: ((), {})
        tensor_specs = {}
        unavailable_reason = ""

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _make_inputs(Config(), device="cuda")


def test_failed_compressor_is_not_replaced_by_svd():
    class BrokenCompressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("constructor failure")

    result = benchmark_single_compressor("broken", BrokenCompressor, device="cpu")
    assert result["status"] == "ERROR"
    assert "constructor failure" in result["error"]
    assert "sqnr_db" not in result


def test_backward_requires_finite_gradients(monkeypatch):
    class DetachedAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, tensor):
            return tensor.detach()

    monkeypatch.setattr(
        "neuralops.benchmarks.legacy_scripts.gpu_master_suite._get_shared_pretrained_attention",
        lambda dim, num_heads: (None, None),
    )
    result = benchmark_single_attention(
        "detached",
        DetachedAttention,
        b=1,
        h=1,
        seq=2,
        d=2,
        device="cpu",
        warmup=0,
        iters=1,
        check_backward=True,
    )
    assert result["backward_passed"] is False
    assert result["status"] == "FAIL_NUMERICAL"


def test_compressor_module_adapter_lifecycle_and_metrics():
    """Verify CompressorModuleAdapter 6-gate lifecycle, snapshotting, and Frobenius/SQNR metrics."""
    assert issubclass(CompressorModuleAdapter, BaseBenchmarkAdapter)

    adapter = CompressorModuleAdapter(
        compressor_name="direct_svd",
        device="cpu",
        seed=42,
        tolerance=1.0,
    )
    result = adapter.execute()

    assert result.status == BenchmarkStatus.SUCCESS.value
    assert adapter._snapshot_model is not None, "Model snapshot must exist before mutation"
    assert adapter.relative_frob_error >= 0.0, "Relative Frobenius error must be non-negative"
    assert "relative_frob_error" in result.metadata
    assert "sqnr_db" in result.metadata
    assert "compression_ratio" in result.metadata
    assert "param_reduction_pct" in result.metadata
    assert result.metadata["relative_frob_error"] == adapter.relative_frob_error
    assert result.metadata["sqnr_db"] == adapter.sqnr_db

    # Verify unified BenchmarkResult interface
    legacy_res = adapter.benchmark_module("direct_svd")
    assert isinstance(legacy_res, BenchmarkResult)
    assert legacy_res.status == BenchmarkStatus.SUCCESS.value
    assert legacy_res.metadata["relative_frob_error"] == adapter.relative_frob_error
    assert legacy_res.metadata["sqnr_db"] == adapter.sqnr_db


def test_compressor_module_adapter_fail_closed_no_svd_substitution(monkeypatch):
    """Verify strict fail-closed: broken compressor fails with FAILED, never generic SVD."""
    class BrokenCompressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("forced compressor surgery abort")

    # In-memory temporary registration for fail-closed verification
    monkeypatch.setitem(COMPRESSOR_REGISTRY._dict, "broken_audit_comp", BrokenCompressor)

    adapter = CompressorModuleAdapter(compressor_name="broken_audit_comp", device="cpu")
    result = adapter.execute()

    assert result.status == BenchmarkStatus.FAILED.value
    assert "forced compressor surgery abort" in (result.error or "")
    assert "sqnr_db" not in result.metadata

    # Verify unregistered compressor fails closed and does not substitute generic SVD
    adapter_unknown = CompressorModuleAdapter(compressor_name="unregistered_failing_xyz", device="cpu")
    res_unknown = adapter_unknown.execute()
    assert res_unknown.status in (BenchmarkStatus.DEPENDENCY_ERROR.value, BenchmarkStatus.FAILED.value)
    assert "unregistered_failing_xyz" in (res_unknown.error or "")
    assert "sqnr_db" not in res_unknown.metadata


def test_compressor_module_adapter_device_fail_closed(monkeypatch):
    """Verify CompressorModuleAdapter fails closed when accelerated device is unavailable."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        CompressorModuleAdapter(compressor_name="direct_svd", device="cuda")
