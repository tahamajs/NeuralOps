"""Tests for eval_pipeline module."""

import torch
from neuralops.evaluation.eval_pipeline import (
    probe_optimal_batch_size,
    evaluate_single_combination_opt,
)
from neuralops.evaluation.eval_patcher import build_model_instance


def test_probe_optimal_batch_size_cpu():
    model, _ = build_model_instance("bert", hidden_size=64, num_layers=1, num_heads=2)
    b = probe_optimal_batch_size(
        model=model,
        model_family="bert",
        dataset="sst2",
        seq_len=16,
        device=torch.device("cpu"),
    )
    assert b == 64


def test_evaluate_single_combination_deprecated_skip():
    res = evaluate_single_combination_opt(
        track="track1_svd",
        module_name="eat",
        module_type="attention",
        model_family="bert",
        dataset="sst2",
        device=torch.device("cpu"),
        skip_deprecated=True,
    )
    assert res.status == "SKIP_DEPRECATED"
