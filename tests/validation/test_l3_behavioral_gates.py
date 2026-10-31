#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Sub-Agent C1: Behavioral Validation Agent (L3 Gates).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from neuralops.validation.gates.behavioral.attention_patterns import AttentionPatternsGate
from neuralops.validation.gates.behavioral.logit_kl import LogitKLGate
from neuralops.validation.gates.behavioral.output_similarity import OutputSimilarityGate
from neuralops.validation.gates.behavioral.perplexity import PerplexityGate


class SimpleMLPNet(nn.Module):
    """Model without attention mechanisms (MLP-only)."""

    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ToyAttentionNet(nn.Module):
    """Model with attention mechanism."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.self_attn(x))


def test_c1_logit_kl_passes_identical_fails_different():
    """Verify logit_kl gate passes on identical models and fails on very different models."""
    m_base = SimpleMLPNet()
    m_identical = copy.deepcopy(m_base)

    # Identical models -> KL divergence ~ 0
    res_identical = LogitKLGate().run(model_before=m_base, model_after=m_identical)
    assert res_identical.passed is True
    assert res_identical.measured_value < 1e-4

    # Very different model (weights perturbed heavily)
    m_different = copy.deepcopy(m_base)
    with torch.no_grad():
        m_different.fc1.weight.add_(torch.randn_like(m_different.fc1.weight) * 10.0)
        m_different.fc2.weight.add_(torch.randn_like(m_different.fc2.weight) * 10.0)

    res_different = LogitKLGate(max_kl=0.01).run(model_before=m_base, model_after=m_different)
    assert res_different.passed is False
    assert res_different.measured_value > 0.01


def test_c1_perplexity_skipped_without_dataset():
    """Verify perplexity gate returns skipped=True if no dataset is provided."""
    m_base = SimpleMLPNet()
    m_after = copy.deepcopy(m_base)

    res = PerplexityGate().run(model_before=m_base, model_after=m_after, context={})
    assert res.skipped is True
    assert res.passed is True
    assert "No evaluation dataset provided" in res.message


def test_c1_output_similarity_works_on_mlp_only():
    """Verify output_similarity gate works on models without attention."""
    m_base = SimpleMLPNet()
    m_after = copy.deepcopy(m_base)

    res = OutputSimilarityGate(min_similarity=0.90).run(model_before=m_base, model_after=m_after)
    assert res.passed is True
    assert res.measured_value >= 0.99


def test_c1_attention_patterns_skipped_on_models_without_attention():
    """Verify attention_patterns gate returns skipped=True on models without attention."""
    m_mlp = SimpleMLPNet()
    m_after = copy.deepcopy(m_mlp)

    res = AttentionPatternsGate().run(model_before=m_mlp, model_after=m_after)
    assert res.skipped is True
    assert res.passed is True
    assert "does not contain recognizable attention mechanisms" in res.message


def test_c1_attention_patterns_runs_on_attention_model():
    """Verify attention_patterns gate runs on model with attention."""
    m_attn = ToyAttentionNet()
    m_after = copy.deepcopy(m_attn)

    res = AttentionPatternsGate().run(model_before=m_attn, model_after=m_after)
    assert res.passed is True
