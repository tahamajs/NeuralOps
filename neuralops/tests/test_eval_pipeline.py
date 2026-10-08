"""Unit tests for neuralops.evaluation.eval_pipeline."""

import pytest
import torch
import torch.nn as nn

from neuralops.evaluation.eval_pipeline import (
    probe_optimal_batch_size,
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
        """Probe should return 64 default when not executed on CUDA."""
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
