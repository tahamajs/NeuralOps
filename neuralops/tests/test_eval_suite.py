"""Tests for Phase 8 evaluation suite."""
import torch
import torch.nn as nn
from neuralops.benchmarks.eval_suite import (
    LMEvalAdapter,
    RULERBenchmark,
    CustomEvaluationSuite,
)


def test_lm_eval_adapter():
    model = nn.Linear(8, 8)
    adapter = LMEvalAdapter(model)
    res = adapter.loglikelihood([("Hello world", "how are you")])
    assert len(res) == 1
    assert res[0][0] < 0.0
    assert res[0][1] is True


def test_ruler_benchmark():
    benchmark = RULERBenchmark(context_lengths=[100, 200])
    needle = "PASSKEY_123"

    def dummy_retriever(prompt: str) -> str:
        if needle in prompt:
            return f"Found {needle} in text."
        return "Not found"

    scores = benchmark.evaluate(dummy_retriever, needle=needle)
    assert scores[100] == 1.0
    assert scores[200] == 1.0


def test_custom_evaluation_suite():
    model = nn.Linear(8, 8)
    suite = CustomEvaluationSuite(model)
    x = torch.randn(2, 8)
    res = suite.evaluate_all(x)
    assert res["parameter_count"] > 0
    assert "output_variance" in res
