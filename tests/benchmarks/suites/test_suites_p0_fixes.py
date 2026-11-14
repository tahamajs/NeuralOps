#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite for P0 benchmark suite fixes:
- roofline.py: honest attention execution, pre-allocated tensors, verified FLOPs/bytes.
- niah.py: raises NotImplementedError without model; runs honest passkey retrieval with model.
- e2e_serving.py: raises NotImplementedError without model; runs honest prefill/decode with model.
- qwen_kd.py: raises NotImplementedError without model; runs honest KD step with model.
- cuda_ops.py: raises NotImplementedError instead of fake lambda mock when kernel is unavailable.
- triton_ops.py: raises ValueError on unsupported kernels instead of fake lambda mock.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.core.base import ExecutionSummary
from neuralops.benchmarks.core.paths import RunDirectoryContext
from neuralops.benchmarks.suites.cuda_ops import CudaOpsBenchmark
from neuralops.benchmarks.suites.e2e_serving import ServingBenchmark
from neuralops.benchmarks.suites.niah import NeedleInAHaystackBenchmark
from neuralops.benchmarks.suites.qwen_kd import QwenKDBenchmark
from neuralops.benchmarks.suites.roofline import RooflineBenchmark
from neuralops.benchmarks.suites.triton_ops import TritonOpsBenchmark
from neuralops.config.benchmark_schema import BenchmarkRunConfig


def _make_dummy_context(tmp_path) -> RunDirectoryContext:
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RunDirectoryContext(
        run_id="test_run_123",
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


class _SyntheticCausalModel(nn.Module):
    def __init__(self, vocab_size: int = 200, dim: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=dim)
        self.embed = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        past_key_values: object = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        pkv = ("cached_state",) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=pkv)


class _SyntheticTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 150 for c in text]


# =========================================================================
# 1. roofline.py tests
# =========================================================================


def test_roofline_honest_execution(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="roofline",
        custom_parameters={"attention_types": ["full", "causal", "mla"], "batch_sizes": [1], "seq_lens": [32]},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = RooflineBenchmark(config, tracker, context)
    suite.setup()

    # Run iteration for full, causal, and mla
    for i, attn_type in enumerate(["full", "causal", "mla"]):
        res = suite.run_iteration(step=i, attention_type=attn_type, batch_size=1, seq_len=32)
        assert "achieved_tflops" in res
        assert "arithmetic_intensity" in res
        assert res["achieved_tflops"] >= 0.0
        assert res["arithmetic_intensity"] > 0.0

    summary = suite.generate_summary()
    assert summary.status == "COMPLETED"
    assert len(summary.results_table) == 3
    for entry in summary.results_table:
        assert entry["status"] == "PASS"
        assert entry["dtype"] == "fp32"


# =========================================================================
# 2. niah.py tests
# =========================================================================


def test_niah_raises_not_implemented_without_model(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="niah",
        custom_parameters={},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = NeedleInAHaystackBenchmark(config, tracker, context)
    with pytest.raises(NotImplementedError, match="requires an actual causal language model"):
        suite.setup()


def test_niah_honest_execution_with_model(tmp_path) -> None:
    model = _SyntheticCausalModel()
    tokenizer = _SyntheticTokenizer()
    config = BenchmarkRunConfig(
        benchmark_suite="niah",
        custom_parameters={
            "model": model,
            "tokenizer": tokenizer,
            "context_lengths": [64],
            "depth_percentages": [0.5],
        },
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = NeedleInAHaystackBenchmark(config, tracker, context)
    suite.setup()
    res = suite.run_iteration(step=0, ctx_len=64, depth=0.5)
    assert "accuracy" in res
    assert "latency_ms" in res
    assert res["latency_ms"] > 0.0


# =========================================================================
# 3. e2e_serving.py tests
# =========================================================================


def test_serving_raises_not_implemented_without_model(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="e2e_serving",
        custom_parameters={},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = ServingBenchmark(config, tracker, context)
    with pytest.raises(NotImplementedError, match="requires an actual model instance"):
        suite.setup()


def test_serving_honest_execution_with_model(tmp_path) -> None:
    model = _SyntheticCausalModel()
    config = BenchmarkRunConfig(
        benchmark_suite="e2e_serving",
        custom_parameters={
            "model": model,
            "batch_sizes": [1],
            "prompt_lens": [16],
            "gen_lens": [4],
        },
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = ServingBenchmark(config, tracker, context)
    suite.setup()
    res = suite.run_iteration(step=0, batch_size=1, prompt_len=16, gen_len=4)
    assert "ttft_ms" in res
    assert "throughput_tokens_per_sec" in res
    assert res["ttft_ms"] > 0.0
    assert res["throughput_tokens_per_sec"] > 0.0


# =========================================================================
# 4. qwen_kd.py tests
# =========================================================================


def test_qwen_kd_raises_not_implemented_without_model(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="qwen_kd",
        custom_parameters={},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = QwenKDBenchmark(config, tracker, context)
    with pytest.raises(NotImplementedError, match="requires actual student and teacher"):
        suite.setup()


def test_qwen_kd_honest_execution_with_models(tmp_path) -> None:
    student = _SyntheticCausalModel()
    teacher = _SyntheticCausalModel()
    config = BenchmarkRunConfig(
        benchmark_suite="qwen_kd",
        custom_parameters={
            "student": student,
            "teacher": teacher,
            "context_length": 32,
        },
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = QwenKDBenchmark(config, tracker, context)
    suite.setup()
    res = suite.run_iteration(step=0, stage="distillation_step")
    assert "kd_loss" in res
    assert "latency_ms" in res
    assert res["latency_ms"] > 0.0
    assert res["kd_loss"] >= 0.0


# =========================================================================
# 5. cuda_ops.py and triton_ops.py tests
# =========================================================================


def test_hardware_ops_raises_when_no_kernel_installed(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="cuda_ops",
        custom_parameters={"kernels": ["quantized_gemm"]},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = CudaOpsBenchmark(config, tracker, context)
    suite.setup()

    with patch("neuralops.benchmarks.suites.cuda_ops.CUDAKernelAdapter") as mock_adapter_cls:
        mock_adapter = MagicMock()
        mock_adapter._cuda_fn = None
        mock_adapter._triton_fn = None
        mock_adapter._pytorch_ref_fn = None
        mock_adapter_cls.return_value = mock_adapter

        # Must record FAILED_RUNTIME with NotImplementedError rather than PASS
        res = suite.run_iteration(step=0, kernel_name="quantized_gemm")
        assert res["cuda_latency_ms"] == 0.0
        assert "FAILED_RUNTIME" in suite.results_table[0]["status"]
        assert "No CUDA, Triton, or PyTorch reference implementation" in suite.results_table[0]["status"]


def test_jit_ops_raises_on_unsupported_kernel(tmp_path) -> None:
    config = BenchmarkRunConfig(
        benchmark_suite="triton_ops",
        custom_parameters={"kernels": ["unknown_op"]},
    )
    context = _make_dummy_context(tmp_path)
    tracker = _DummyTracker()

    suite = TritonOpsBenchmark(config, tracker, context)
    suite.setup()

    with pytest.raises(ValueError, match="Unsupported Triton kernel"):
        suite._prepare_inputs("unknown_op")
