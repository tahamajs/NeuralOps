"""Unit tests for neuralops.evaluation.eval_pipeline."""
from __future__ import annotations

import time
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
        env = _setup_eval_environment(
            model_family="bert",
            dataset="sst2",
            seq_len=32,
            device=torch.device("cpu"),
        )
        assert isinstance(env, dict)
        assert "model" in env
        assert "cfg" in env


class TestFinalizeBenchmarkResult:
    def test_finalize_pass(self, tmp_path):
        """Test finalizing a successful benchmark result."""
        base_res = BenchmarkResult(
            timestamp="2026-10-08T00:00:00",
            track="track1",
            module_name="test_mod",
            module_type="compressor",
            model_family="bert",
            dataset="sst2",
            status="PASS",
            accuracy_pct=92.5,
            loss=0.3,
            compression_ratio=0.75,
        )
        res = _finalize_benchmark_result(
            result=base_res,
            run_log_lines=[],
            t_run_start=time.time(),
            module_name="test_mod",
            module_type="compressor",
            model_family="bert",
            dataset="sst2",
            rank=16,
            seq_len=32,
            batch_size=16,
            current_batch_size=16,
            num_warmup=2,
            num_steps=5,
            allow_synthetic=True,
            device=torch.device("cpu"),
            results_dir=str(tmp_path),
            comp_stats={"rank": 16},
            mem_phases={"peak_mb": {"allocated_mb": 128.0}},
            t_model_load=0.1,
            t_calib=0.05,
            t_comp=0.2,
            t_eval=0.3,
            baseline_logits_samples=[[0.1, 0.2]],
            compressed_logits_samples=[[0.1, 0.2]],
            sample_cosines=[0.99, 0.98],
            exc_traceback=None,
            exc_type=None,
            retry_count=0,
            success=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "PASS"
        assert res.module_name == "test_mod"
        assert res.model_family == "bert"

    def test_finalize_failure(self, tmp_path):
        """Test finalizing an errored benchmark result."""
        base_res = BenchmarkResult(
            timestamp="2026-10-08T00:00:00",
            track="track2",
            module_name="broken_mod",
            module_type="attention",
            model_family="gpt2",
            dataset="wikitext2",
            status="FAIL",
        )
        res = _finalize_benchmark_result(
            result=base_res,
            run_log_lines=[],
            t_run_start=time.time(),
            module_name="broken_mod",
            module_type="attention",
            model_family="gpt2",
            dataset="wikitext2",
            rank=16,
            seq_len=64,
            batch_size=8,
            current_batch_size=8,
            num_warmup=0,
            num_steps=0,
            allow_synthetic=True,
            device=torch.device("cpu"),
            results_dir=str(tmp_path),
            comp_stats={},
            mem_phases={},
            t_model_load=0.1,
            t_calib=0.0,
            t_comp=0.0,
            t_eval=0.0,
            baseline_logits_samples=[],
            compressed_logits_samples=[],
            sample_cosines=[],
            exc_traceback="ValueError: Invalid config",
            exc_type="ValueError",
            retry_count=1,
            success=False,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAIL"


class TestEvaluateSingleCombination:
    def test_evaluate_single_combination_alias(self):
        """Verify alias points to optimize implementation."""
        assert evaluate_single_combination is evaluate_single_combination_opt

    def test_evaluate_error_handling_invalid_model(self, tmp_path):
        """Should catch invalid model family gracefully and return BenchmarkResult with FAIL status."""
        res = evaluate_single_combination_opt(
            track="track1",
            module_name="svd_linear_attention",
            module_type="attention",
            model_family="nonexistent_model_family_123",
            dataset="sst2",
            results_dir=str(tmp_path),
            device=torch.device("cpu"),
            allow_synthetic=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAIL"

    def test_evaluate_error_handling_invalid_module(self, tmp_path):
        """Should catch unknown module gracefully."""
        res = evaluate_single_combination_opt(
            track="track1",
            module_name="nonexistent_attention_module_xyz",
            module_type="attention",
            model_family="bert",
            dataset="sst2",
            results_dir=str(tmp_path),
            device=torch.device("cpu"),
            allow_synthetic=True,
        )
        assert isinstance(res, BenchmarkResult)
        assert res.status == "FAIL"


class TestBenchmarkResultDataclass:
    def test_benchmark_result_to_dict(self):
        result = BenchmarkResult(
            timestamp="2026-10-08T00:00:00",
            track="track1",
            module_name="direct_svd",
            module_type="compressor",
            model_family="bert",
            dataset="sst2",
            status="PASS",
            accuracy_pct=92.5,
            loss=0.32,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["module_name"] == "direct_svd"
        assert d["accuracy_pct"] == 92.5
        assert d["loss"] == 0.32
