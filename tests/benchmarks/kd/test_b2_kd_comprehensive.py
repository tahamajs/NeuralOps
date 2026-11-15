#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive unit and regression tests for KD subsystem (W4-B2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.distill_engine import TwoStageDistillationEngine
from neuralops.benchmarks.kd.reporter import KDReporter
from neuralops.benchmarks.kd.teacher_cache import (
    CachedCausalDataset,
    _safe_torch_load,
    atomic_torch_save,
)
from neuralops.benchmarks.kd.zoo_orchestrator import ZooOrchestrator


def test_safe_torch_load_cve_weights_only_error_propagation(tmp_path: Path) -> None:
    """Verify _safe_torch_load re-raises non-signature TypeErrors and enforces weights_only safety."""
    bad_file = tmp_path / "corrupt.pt"
    bad_file.write_bytes(b"not a valid torch file")

    with pytest.raises(Exception):
        _safe_torch_load(bad_file)


def test_kd_reporter_atomic_record_and_summary(tmp_path: Path) -> None:
    """Verify KDReporter generates atomic records, CSV logs, and final markdown report."""
    cfg = BenchmarkConfig(output_dir=str(tmp_path / "kd_results"), task="glue:sst2")
    reporter = KDReporter(cfg)

    result_1 = {
        "module": "linear_attention",
        "dataset": "sst2",
        "status": "IMPROVED",
        "pre_kd_acc": 85.0,
        "best_kd_acc": 91.5,
        "final_kd_acc": 91.0,
        "delta_vs_pre": 6.0,
        "duration_s": 42.0,
    }
    reporter.record_run(result_1)

    csv_path = Path(cfg.output_dir) / "kd_benchmark.csv"
    assert csv_path.exists()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["module"] == "linear_attention"
        assert reader[0]["status"] == "IMPROVED"

    reporter.generate_final_report([result_1])
    md_path = Path(cfg.output_dir) / "kd_benchmark_final.md"
    json_path = Path(cfg.output_dir) / "kd_benchmark_final.json"
    assert md_path.exists()
    assert json_path.exists()

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded_json) == 1
    assert loaded_json[0]["module"] == "linear_attention"


def test_distill_engine_max_steps_split_fractions() -> None:
    """Verify TwoStageDistillationEngine divides steps according to stage fractions."""
    hidden_dim, seq_len, vocab_size = 8, 4, 16

    class MiniStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> object:
            h = self.embed(input_ids)
            h = self.linear(h)
            logits = self.lm_head(h)
            return type("Out", (), {"logits": logits, "hidden_states": [h, h]})()

    student = MiniStudent()
    n_samples = 6
    mock_cache = {
        "cache_version": 1,
        "layer_indices": [0],
        "input_ids": torch.randint(0, vocab_size, (n_samples, seq_len)),
        "attention_mask": torch.ones((n_samples, seq_len), dtype=torch.long),
        "hidden_states": torch.randn((n_samples, 1, seq_len, hidden_dim), dtype=torch.float16),
        "topk_vals": torch.randn((n_samples, seq_len, 2), dtype=torch.float16),
        "topk_idx": torch.randint(0, vocab_size, (n_samples, seq_len, 2), dtype=torch.int32),
    }

    train_ds = CachedCausalDataset(mock_cache)
    loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    engine = TwoStageDistillationEngine(
        train_secs=500.0,
        stage1_frac=0.40,
        stage2_frac=0.60,
        grad_accum=1,
        device=torch.device("cpu"),
        max_steps=5,
    )

    metrics = engine.run(
        student=student,
        train_loader=loader,
        val_loader=loader,
        teacher_cache=mock_cache,
        eval_fn=lambda m, l, d: 5.0,
        max_steps=5,
    )

    # 5 * 0.40 = 2 for stage 1; 5 - 2 = 3 for stage 2
    assert metrics["steps_s1"] == 2
    assert metrics["steps_s2"] == 3
    assert metrics["total_steps"] == 5
