"""Tests for eval_patcher module."""

import torch
from neuralops.evaluation.eval_patcher import (
    build_model_instance,
    get_base_model_cached,
    apply_module_to_model,
)


def test_build_model_instance():
    model, cfg = build_model_instance("bert", hidden_size=64, num_layers=1, num_heads=2)
    assert model is not None
    assert cfg.hidden_size == 64


def test_get_base_model_cached():
    m1, _ = get_base_model_cached("bert", hidden_size=64, num_layers=1, num_heads=2)
    m2, _ = get_base_model_cached("bert", hidden_size=64, num_layers=1, num_heads=2)
    assert m1 is not m2  # isolated fresh instances
    assert sum(p.numel() for p in m1.parameters()) == sum(p.numel() for p in m2.parameters())


def test_apply_module_to_model_baseline():
    m, _ = build_model_instance("bert", hidden_size=64, num_layers=1, num_heads=2)
    m_patched, ratio = apply_module_to_model(m, "baseline")
    assert ratio == 1.0
