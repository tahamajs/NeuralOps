#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for RULER benchmark evaluator (neuralops.evaluation.ruler)."""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralops.evaluation.eval_patcher import build_model_instance
from neuralops.evaluation.ruler import RulerEvaluator, evaluate_ruler, SUPPORTED_RULER_TASKS


class MockTokenizer:
    """Lightweight deterministic tokenizer for testing without network calls."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._vocab: dict[str, int] = {}
        self._rev_vocab: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens: list[int] = []
        for word in text.split():
            clean = word.strip()
            if clean not in self._vocab:
                idx = len(self._vocab) + 1
                self._vocab[clean] = idx
                self._rev_vocab[idx] = clean
            tokens.append(self._vocab[clean])
        return tokens or [1]

    def decode(self, tokens: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        words = [self._rev_vocab.get(int(t), str(t)) for t in tokens]
        return " ".join(words)


def test_ruler_supported_tasks():
    """Verify supported tasks contract."""
    assert "single_needle" in SUPPORTED_RULER_TASKS
    assert "multi_needle" in SUPPORTED_RULER_TASKS
    assert "variable_tracking" in SUPPORTED_RULER_TASKS


def test_ruler_task_generation():
    """Verify each RULER task synthesizes valid tokens and expected answers."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    evaluator = RulerEvaluator(model, tok, context_lengths=[64], num_samples_per_task=1)

    toks_single, ans_single = evaluator._generate_single_needle_task(64, depth=0.5)
    assert len(toks_single) > 10
    assert isinstance(ans_single, str)
    assert len(ans_single) == 5  # 5-digit security code

    toks_multi, ans_multi = evaluator._generate_multi_needle_task(64, depth=0.5)
    assert len(toks_multi) > 10
    assert isinstance(ans_multi, str)

    toks_var, ans_var = evaluator._generate_variable_tracking_task(64, depth=0.5)
    assert len(toks_var) > 10
    assert isinstance(ans_var, str)


def test_ruler_evaluator_execution():
    """Verify end-to-end evaluation produces correct result schema."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    evaluator = RulerEvaluator(
        model,
        tok,
        context_lengths=[32],
        tasks=["single_needle"],
        num_samples_per_task=1,
        device="cpu",
        seed=123,
    )

    results = evaluator.evaluate()
    assert "accuracy_grid" in results
    assert "mean_accuracy" in results
    assert "task_accuracy" in results
    assert "context_accuracy" in results
    assert "records" in results
    assert len(results["records"]) == 1
    rec = results["records"][0]
    assert rec["task"] == "single_needle"
    assert "expected" in rec
    assert "prediction" in rec


def test_evaluate_ruler_helper():
    """Verify evaluate_ruler convenience function."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    results = evaluate_ruler(
        model,
        tok,
        context_lengths=[32],
        tasks=["variable_tracking"],
        num_samples_per_task=1,
        device="cpu",
    )
    assert results["mean_accuracy"] >= 0.0
    assert "variable_tracking" in results["task_accuracy"]
