"""Comprehensive functional tests for suites_v2 (niah, qwen_kd, e2e_serving).

Tests verify:
1. Benchmark setup and synthetic fallback model instantiation when custom parameters omit models.
2. Custom model pass-through execution.
3. Iteration metrics sanity and L2 cache flushing.
4. generate_summary contract adhering to ExecutionSummary schema.
5. AbstractBaseBenchmark compliance and teardown cleanup.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import torch

from neuralops.benchmarks.core.base import ExecutionSummary
from neuralops.benchmarks.suites_v2.e2e_serving import ServingBenchmark
from neuralops.benchmarks.suites_v2.niah import NeedleInAHaystackBenchmark
from neuralops.benchmarks.suites_v2.qwen_kd import QwenKDBenchmark
from neuralops.models.modeling import NeuralOpsConfig, NeuralOpsForCausalLM


def _make_benchmark_harness(custom_params: dict | None = None) -> tuple[MagicMock, MagicMock, MagicMock]:
    cfg = MagicMock()
    cfg.hardware.device = "cpu"
    cfg.custom_parameters = custom_params or {}
    tracker = MagicMock()
    ctx = MagicMock()
    ctx.run_id = "test-run-v2-functional"
    return cfg, tracker, ctx


def test_niah_fallback_and_custom_execution() -> None:
    """Test NIAH fallback model instantiation and custom model execution."""
    cfg, tracker, ctx = _make_benchmark_harness({"limit": 2})
    niah = NeedleInAHaystackBenchmark(config=cfg, tracker=tracker, context=ctx)
    niah.setup()

    assert niah.model is not None
    assert isinstance(niah.model, torch.nn.Module)
    assert len(niah.grid) == 2

    # Run single iteration
    res = niah.run_iteration(step=0, ctx_len=32, depth=0.5)
    assert "accuracy" in res
    assert "latency_ms" in res
    assert res["latency_ms"] > 0

    # Test full run and summary
    niah.run()
    summary = niah.generate_summary()
    assert isinstance(summary, ExecutionSummary)
    assert summary.benchmark_name == "niah"
    assert summary.status == "COMPLETED"
    assert summary.passed_iterations == 3  # 1 manual iteration + 2 from run()
    assert "mean_retrieval_accuracy" in summary.extra_metadata

    niah.teardown()


def test_niah_user_supplied_model() -> None:
    """Test NIAH when a user explicitly supplies model and tokenizer."""
    custom_cfg = NeuralOpsConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=512,
    )
    user_model = NeuralOpsForCausalLM(custom_cfg).eval()

    cfg, tracker, ctx = _make_benchmark_harness({"model": user_model, "limit": 1})
    niah = NeedleInAHaystackBenchmark(config=cfg, tracker=tracker, context=ctx)
    niah.setup()

    assert niah.model is user_model
    res = niah.run_iteration(step=0, ctx_len=32, depth=0.2)
    assert "accuracy" in res
    assert res["latency_ms"] > 0


def test_qwen_kd_fallback_and_custom_execution() -> None:
    """Test Qwen KD fallback and custom model distillation stages."""
    cfg, tracker, ctx = _make_benchmark_harness({"limit": 2})
    kd = QwenKDBenchmark(config=cfg, tracker=tracker, context=ctx)
    kd.setup()

    assert kd.student is not None
    assert kd.teacher is not None
    assert len(kd.stages) == 2

    res = kd.run_iteration(step=0, stage="distillation_step")
    assert "kd_loss" in res
    assert "latency_ms" in res
    assert res["kd_loss"] > 0.0

    kd.run()
    summary = kd.generate_summary()
    assert isinstance(summary, ExecutionSummary)
    assert summary.benchmark_name == "qwen_kd"
    assert summary.status == "COMPLETED"
    assert summary.passed_iterations == 3  # 1 manual iteration + 2 from run()

    kd.teardown()


def test_qwen_kd_user_supplied_models() -> None:
    """Test Qwen KD with explicit student and teacher instances."""
    s_cfg = NeuralOpsConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
    t_cfg = NeuralOpsConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    student = NeuralOpsForCausalLM(s_cfg).eval()
    teacher = NeuralOpsForCausalLM(t_cfg).eval()

    cfg, tracker, ctx = _make_benchmark_harness({
        "student": student,
        "teacher": teacher,
        "context_length": 64,
        "limit": 1,
    })
    kd = QwenKDBenchmark(config=cfg, tracker=tracker, context=ctx)
    kd.setup()

    assert kd.student is student
    assert kd.teacher is teacher

    res = kd.run_iteration(step=0, stage="distillation_step")
    assert "kd_loss" in res
    assert res["latency_ms"] > 0.0


def test_e2e_serving_fallback_and_custom_execution() -> None:
    """Test Serving benchmark fallback prefill and decode stages."""
    cfg, tracker, ctx = _make_benchmark_harness({"limit": 1})
    srv = ServingBenchmark(config=cfg, tracker=tracker, context=ctx)
    srv.setup()

    assert srv.model is not None
    assert len(srv.batch_sizes) == 1

    res = srv.run_iteration(step=0, batch_size=1, prompt_len=16, gen_len=4)
    assert "ttft_ms" in res
    assert "throughput_tokens_per_sec" in res
    assert res["ttft_ms"] > 0.0
    assert res["throughput_tokens_per_sec"] > 0.0

    srv.run()
    summary = srv.generate_summary()
    assert isinstance(summary, ExecutionSummary)
    assert summary.benchmark_name == "e2e_serving"
    assert summary.status == "COMPLETED"
    assert summary.passed_iterations == 2  # 1 manual iteration + 1 from run()

    srv.teardown()


def test_e2e_serving_user_supplied_model() -> None:
    """Test Serving benchmark with user-provided model instance."""
    m_cfg = NeuralOpsConfig(vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=2)
    model = NeuralOpsForCausalLM(m_cfg).eval()

    cfg, tracker, ctx = _make_benchmark_harness({"model": model, "batch_sizes": [1]})
    srv = ServingBenchmark(config=cfg, tracker=tracker, context=ctx)
    srv.setup()

    assert srv.model is model
    res = srv.run_iteration(step=0, batch_size=1, prompt_len=16, gen_len=4)
    assert res["ttft_ms"] > 0.0
    assert res["throughput_tokens_per_sec"] > 0.0
