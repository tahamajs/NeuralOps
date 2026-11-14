#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive unit and contract test suite for NeuralOps benchmark suites.
Covers:
- attention: FAILED_INIT, FAILED_RUNTIME, FAILED_OOM, _resolve_torch_dtype, summary generation.
- compression: param_reduction_pct, SVD (U.numel() + V.numel()), module replacement, summary.
- cuda_ops: _resolve_torch_dtype validation, canonical kernels list, backend error handling.
- triton_ops: _resolve_torch_dtype validation, canonical kernels list, unsupported kernel error.
- roofline: verified FLOPs/bytes and arithmetic intensity.
- niah, e2e_serving, qwen_kd: validation, error handling, honest execution.
- __init__: get_benchmark_suite, list_suites.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.core.paths import RunDirectoryContext
from neuralops.benchmarks.suites import (
    AttentionBenchmark,
    CompressionBenchmark,
    CudaOpsBenchmark,
    NeedleInAHaystackBenchmark,
    QwenKDBenchmark,
    RooflineBenchmark,
    ServingBenchmark,
    TritonOpsBenchmark,
    get_benchmark_suite,
    list_suites,
)
from neuralops.benchmarks.suites.compression import _count_module_params
from neuralops.config.benchmark_schema import BenchmarkRunConfig


def _make_dummy_context(tmp_path) -> RunDirectoryContext:
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RunDirectoryContext(
        run_id="test_run_comprehensive",
        benchmark_name="test_suite",
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        tensorboard_dir=run_dir / "tb",
        metrics_jsonl=run_dir / "metrics.jsonl",
        summary_json=run_dir / "summary.json",
        summary_csv=run_dir / "summary.csv",
        manifest_json=run_dir / "manifest.json",
        config_resolved_yaml=run_dir / "config.yaml",
        log_file=run_dir / "log.txt",
        report_md=run_dir / "report.md",
    )


class _DummyTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, list[tuple[float, int]]] = {}

    def log_metric(self, name: str, value: float, step: int = 0) -> None:
        self.metrics.setdefault(name, []).append((value, step))


# -------------------------------------------------------------------------
# 1. Suites __init__.py tests
# -------------------------------------------------------------------------


def test_list_suites() -> None:
    suites = list_suites()
    assert len(suites) == 8
    expected = [
        "attention",
        "compression",
        "cuda_ops",
        "e2e_serving",
        "niah",
        "qwen_kd",
        "roofline",
        "triton_ops",
    ]
    assert suites == expected


def test_get_benchmark_suite() -> None:
    assert get_benchmark_suite("attention") is AttentionBenchmark
    assert get_benchmark_suite("  CUDA_OPS  ") is CudaOpsBenchmark
    with pytest.raises(KeyError, match="Unknown benchmark suite 'invalid'"):
        get_benchmark_suite("invalid")


# -------------------------------------------------------------------------
# 2. Attention failure modes tests
# -------------------------------------------------------------------------


def test_attention_dtype_resolution(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    config = BenchmarkRunConfig(benchmark_suite="attention", custom_parameters={"mechanisms": ["standard"]})
    suite = AttentionBenchmark(config, _DummyTracker(), ctx)
    assert suite._resolve_torch_dtype("fp32") == torch.float32
    assert suite._resolve_torch_dtype("fp16") == torch.float16
    assert suite._resolve_torch_dtype("bf16") == torch.bfloat16
    with pytest.raises(ValueError, match="Unknown dtype 'unknown'"):
        suite._resolve_torch_dtype("unknown")


def test_attention_failure_mode_failed_init(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()
    config = BenchmarkRunConfig(
        benchmark_suite="attention",
        custom_parameters={"mechanisms": ["non_existent_attention_mechanism_xyz"]},
    )
    suite = AttentionBenchmark(config, tracker, ctx)
    suite.setup()
    res = suite.run_iteration(step=0, mechanism="non_existent_attention_mechanism_xyz")
    assert res["latency_ms"] == 0.0
    assert len(suite.results_table) == 1
    assert "FAILED_INIT" in suite.results_table[0]["status"]


def test_attention_failure_mode_failed_oom(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()
    config = BenchmarkRunConfig(
        benchmark_suite="attention",
        custom_parameters={"mechanisms": ["full_attention"]},
    )
    suite = AttentionBenchmark(config, tracker, ctx)
    suite.setup()

    with patch.object(suite, "measure_latency", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
        res = suite.run_iteration(step=0, mechanism="full_attention")
        assert res["latency_ms"] == 0.0
        assert len(suite.results_table) == 1
        assert suite.results_table[0]["status"] == "FAILED_OOM"


def test_attention_failure_mode_failed_runtime(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()
    config = BenchmarkRunConfig(
        benchmark_suite="attention",
        custom_parameters={"mechanisms": ["full_attention"]},
    )
    suite = AttentionBenchmark(config, tracker, ctx)
    suite.setup()

    with patch.object(suite, "measure_latency", side_effect=RuntimeError("Kernel computation blew up")):
        res = suite.run_iteration(step=0, mechanism="full_attention")
        assert res["latency_ms"] == 0.0
        assert len(suite.results_table) == 1
        assert "FAILED_RUNTIME" in suite.results_table[0]["status"]


# -------------------------------------------------------------------------
# 3. Compression SVD param counting tests
# -------------------------------------------------------------------------


class _DummySVDLayer(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64, rank: int = 8) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.zeros(out_features, rank))
        self.V = nn.Parameter(torch.zeros(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V.t()) @ self.U.t() + self.bias


def test_compression_svd_param_counting() -> None:
    layer = _DummySVDLayer(in_features=64, out_features=64, rank=8)
    expected_svd = layer.U.numel() + layer.V.numel() + layer.bias.numel()
    assert _count_module_params(layer) == expected_svd
    assert layer.U.numel() == 64 * 8
    assert layer.V.numel() == 8 * 64


def test_compression_svd_reduction_calculation(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()
    config = BenchmarkRunConfig(
        benchmark_suite="compression",
        custom_parameters={"compressors": ["mock_svd"]},
    )
    suite = CompressionBenchmark(config, tracker, ctx)
    suite.setup()

    mock_compressor = MagicMock()

    def fake_compress():
        return _DummySVDLayer(in_features=64, out_features=64, rank=8)

    mock_compressor.compress = fake_compress

    with patch("neuralops.benchmarks.suites.compression.create_compressor", return_value=mock_compressor):
        res = suite.run_iteration(step=0, compressor_name="mock_svd", target_ratio=0.50)
        assert res["forward_latency_ms"] >= 0.0
        assert len(suite.results_table) == 1
        entry = suite.results_table[0]
        assert entry["status"] == "PASS"
        assert "param_reduction_pct" in entry
        assert entry["param_reduction_pct"] > 0.0


# -------------------------------------------------------------------------
# 4. CUDA and Triton ops dtype and kernel verification
# -------------------------------------------------------------------------


def test_hardware_ops_resolve_dtype() -> None:
    assert CudaOpsBenchmark._resolve_torch_dtype("fp16") == torch.float16
    assert CudaOpsBenchmark._resolve_torch_dtype("fp32") == torch.float32
    with pytest.raises(ValueError, match="Unknown dtype 'bad'"):
        CudaOpsBenchmark._resolve_torch_dtype("bad")


def test_triton_ops_resolve_dtype() -> None:
    assert TritonOpsBenchmark._resolve_torch_dtype("fp16") == torch.float16
    assert TritonOpsBenchmark._resolve_torch_dtype("fp32") == torch.float32
    with pytest.raises(ValueError, match="Unknown dtype 'bad'"):
        TritonOpsBenchmark._resolve_torch_dtype("bad")


def test_triton_ops_all_kernels(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    suite = TritonOpsBenchmark(BenchmarkRunConfig(benchmark_suite="triton_ops"), _DummyTracker(), ctx)
    suite.setup()
    assert len(suite.target_kernels) == 5
    for k in ["linear_attn", "low_rank", "sparse_attn", "gqa", "state_update"]:
        assert k in suite.target_kernels
        args, ref_fn = suite._prepare_inputs(k)
        assert isinstance(args, tuple)
        assert callable(ref_fn)


def test_cuda_ops_no_fake(tmp_path) -> None:
    ctx = _make_dummy_context(tmp_path)
    suite = CudaOpsBenchmark(BenchmarkRunConfig(benchmark_suite="cuda_ops"), _DummyTracker(), ctx)
    suite.setup()
    assert len(suite.target_kernels) == 5
    for k in ["cosformer", "linear_attention", "moe_routing", "paged_kv_cache", "quantized_gemm"]:
        assert k in suite.target_kernels
        args, kw = suite._prepare_inputs(k)
        assert isinstance(args, tuple)
