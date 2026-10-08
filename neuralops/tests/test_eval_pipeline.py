"""Unit tests for neuralops.evaluation.eval_pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from neuralops.evaluation.eval_pipeline import (
    probe_optimal_batch_size,
    _setup_eval_environment,
    _finalize_benchmark_result,
    evaluate_single_combination_opt,
    evaluate_single_combination,
)
from neuralops.evaluation.eval_types import BenchmarkResult


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 2)

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            logits = torch.randn(batch_size, 2)
        else:
            logits = torch.randn(2, 2)
        return type("Output", (), {"logits": logits, "loss": torch.tensor(0.5)})()


class TestProbeOptimalBatchSize:
    def test_probe_with_cpu(self):
        """Probe should return 64 default when executed on CPU."""
        model = DummyModel()
        res = probe_optimal_batch_size(
            model=model,
            model_family="bert",
            dataset="sst2",
            seq_len=32,
            device=torch.device("cpu"),
            allow_synthetic=True,
        )
        assert isinstance(res, int)
        assert res == 64

    def test_probe_handles_exception_fallback(self):
        """Probe falls back safely if forward execution errors."""
        model = DummyModel()
        with patch.object(model, "forward", side_effect=RuntimeError("OOM")):
            res = probe_optimal_batch_size(
                model=model,
                model_family="bert",
                dataset="sst2",
                seq_len=32,
                device=torch.device("cpu"),
                allow_synthetic=True,
            )
            assert res == 64


class TestSetupEvalEnvironment:
    def test_setup_eval_env_cpu(self):
        """Test environment setup with CPU device."""
        device, effective_dtype = _setup_eval_environment("cpu", "float32")
        assert device == torch.device("cpu")
        assert effective_dtype == torch.float32


class TestFinalizeBenchmarkResult:
    def test_finalize_success(self, tmp_path):
        """Test finalizing a successful benchmark result."""
        res = _finalize_benchmark_result(
            model_family="bert",
            module_name="test_mod",
            module_type="compressor",
            dataset="sst2",
            track="track1",
            batch_size=16,
            seq_len=32,
            num_warmup=2,
            num_steps=5,
            allow_synthetic=True,
            device=torch.device("cpu"),
            results_dir=tmp_path,
            comp_stats={"rank": 16},
            mem_phases={"peak_mb": 128.0},
            t_model_load=0.1,
            t_calib=0.05,
            t_comp=0.2,
            t_eval=0.3,
            baseline_logits_samples=torch.randn(4, 2),
            compressed_logits_samples=torch.randn(4, 2),
            sample_cosines=[0.99, 0.98],
            exc_traceback="",
            exc_type="",
            retry_count=0,
            success=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "SUCCESS"
        assert res.module_name == "test_mod"
        assert res.model_family == "bert"

    def test_finalize_failure(self, tmp_path):
        """Test finalizing an errored benchmark result."""
        res = _finalize_benchmark_result(
            model_family="gpt2",
            module_name="broken_mod",
            module_type="attention",
            dataset="wikitext2",
            track="track2",
            batch_size=8,
            seq_len=64,
            num_warmup=0,
            num_steps=0,
            allow_synthetic=True,
            device=torch.device("cpu"),
            results_dir=tmp_path,
            comp_stats={},
            mem_phases={},
            t_model_load=0.1,
            t_calib=0.0,
            t_comp=0.0,
            t_eval=0.0,
            baseline_logits_samples=None,
            compressed_logits_samples=None,
            sample_cosines=[],
            exc_traceback="ValueError: Invalid config",
            exc_type="ValueError",
            retry_count=1,
            success=False,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAILED"
        assert res.error_type == "ValueError"


class TestEvaluateSingleCombination:
    def test_evaluate_single_combination_alias(self):
        """Verify alias points to optimize implementation."""
        assert evaluate_single_combination is evaluate_single_combination_opt

    def test_evaluate_error_handling_invalid_model(self, tmp_path):
        """Should catch invalid model family gracefully and return BenchmarkResult with FAILED status."""
        res = evaluate_single_combination_opt(
            model_family="nonexistent_model_family_123",
            module_name="svd_linear_attention",
            module_type="attention",
            dataset="sst2",
            results_dir=tmp_path,
            device_str="cpu",
            allow_synthetic=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAILED"
        assert res.error_type != ""

    def test_evaluate_error_handling_invalid_module(self, tmp_path):
        """Should catch unknown module gracefully."""
        res = evaluate_single_combination_opt(
            model_family="bert",
            module_name="nonexistent_attention_module_xyz",
            module_type="attention",
            dataset="sst2",
            results_dir=tmp_path,
            device_str="cpu",
            allow_synthetic=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAILED"


class TestBenchmarkResultDataclass:
    def test_benchmark_result_to_dict(self):
        result = BenchmarkResult(
            timestamp="2026-10-08T00:00:00",
            track="track1",
            module_name="direct_svd",
            module_type="compressor",
            model_family="bert",
            dataset="sst2",
            status="SUCCESS",
            accuracy_pct=92.5,
            loss=0.32,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["module_name"] == "direct_svd"
        assert d["accuracy_pct"] == 92.5
        assert d["loss"] == 0.32
