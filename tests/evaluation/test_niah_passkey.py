#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for NIAH Passkey Evaluator (neuralops.evaluation.niah_passkey)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import torch
import torch.nn as nn

from neuralops.evaluation.eval_patcher import build_model_instance
from neuralops.evaluation.niah_passkey import (
    NeedleInAHaystackEvaluator,
    plot_niah_heatmap,
    save_niah_results,
)


class MockTokenizer:
    """Deterministic in-memory tokenizer for unit testing without HuggingFace Hub."""

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


def test_niah_evaluator_init():
    """Verify evaluator initializes with custom context lengths and depths."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    evaluator = NeedleInAHaystackEvaluator(
        model=model,
        tokenizer=tok,
        context_lengths=[128, 256],
        needle_depths=[0.2, 0.8],
        passkey_digits=4,
        device="cpu",
        seed=999,
    )
    assert evaluator.context_lengths == [128, 256]
    assert evaluator.needle_depths == [0.2, 0.8]
    assert evaluator.passkey_digits == 4
    assert evaluator.seed == 999
    assert str(evaluator.device) == "cpu"


def test_niah_passkey_generation():
    """Verify passkey length matches digits parameter."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    evaluator = NeedleInAHaystackEvaluator(model, tok, passkey_digits=6, device="cpu")
    key = evaluator._generate_passkey()
    assert len(key) == 6
    assert key.isdigit()


def test_niah_evaluate_grid():
    """Verify grid evaluation produces numerical accuracy matrix and records."""
    model, _ = build_model_instance("gpt2", num_labels=0, hidden_size=64, num_layers=1, num_heads=2)
    tok = MockTokenizer()
    evaluator = NeedleInAHaystackEvaluator(
        model=model,
        tokenizer=tok,
        context_lengths=[64],
        needle_depths=[0.5],
        passkey_digits=4,
        device="cpu",
        seed=42,
    )
    results = evaluator.evaluate(num_trials_per_cell=1)

    assert "accuracy_grid" in results
    assert len(results["accuracy_grid"]) == 1
    assert len(results["accuracy_grid"][0]) == 1
    assert "mean_retrieval_accuracy" in results
    assert 0.0 <= results["mean_retrieval_accuracy"] <= 1.0
    assert "seed" in results
    assert len(results["records"]) == 1

    rec = results["records"][0]
    assert rec["context_length"] == 64
    assert rec["depth"] == 0.5
    assert len(rec["passkey"]) == 4
    assert isinstance(rec["correct"], bool)


def test_save_niah_results(tmp_path: Path):
    """Verify save_niah_results produces valid JSON and CSV files."""
    dummy_results = {
        "context_lengths": [128],
        "needle_depths": [0.5],
        "accuracy_grid": [[1.0]],
        "mean_retrieval_accuracy": 1.0,
        "records": [
            {
                "context_length": 128,
                "depth": 0.5,
                "passkey": "12345",
                "prediction": "The passkey is 12345",
                "correct": True,
            }
        ],
    }

    json_file = tmp_path / "niah_test.json"
    csv_file = tmp_path / "niah_test.csv"

    save_niah_results(dummy_results, json_path=json_file, csv_path=csv_file)

    assert json_file.is_file()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["mean_retrieval_accuracy"] == 1.0

    assert csv_file.is_file()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["passkey"] == "12345"
    assert reader[0]["correct"] == "True"


def test_plot_niah_heatmap(tmp_path: Path):
    """Verify plot_niah_heatmap saves image output."""
    dummy_results = {
        "context_lengths": [128, 256],
        "needle_depths": [0.25, 0.75],
        "accuracy_grid": [[1.0, 0.8], [0.9, 0.7]],
        "mean_retrieval_accuracy": 0.85,
    }
    img_file = tmp_path / "heatmap.png"
    plot_niah_heatmap(dummy_results, save_path=img_file)
    assert img_file.is_file()
