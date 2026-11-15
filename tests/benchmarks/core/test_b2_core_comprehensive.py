#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive unit and regression tests for Benchmarks Core subsystem (W4-B2)."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary
from neuralops.benchmarks.core.evaluation import (
    evaluate_classification,
    evaluate_perplexity,
    get_memory_fragmentation,
    measure_resources,
)
from neuralops.benchmarks.core.kd import KDLoss, build_teacher_cache, kd_train_step
from neuralops.benchmarks.core.model_loader import apply_attention_module, apply_compressor
from neuralops.benchmarks.core.paths import StoragePathManager
from neuralops.benchmarks.core.stats import bootstrap_ci, fdr_correction, paired_ttest
from neuralops.benchmarks.core.training import finetune_classification


class _DummyBenchmark(AbstractBaseBenchmark):
    def setup(self) -> None:
        pass

    def run_iteration(self, step: int, **kwargs: object) -> dict[str, float]:
        return {"step": float(step)}

    def teardown(self) -> None:
        pass

    def generate_summary(self) -> ExecutionSummary:
        return ExecutionSummary(run_id="test_run", benchmark_name="test_bm")

    def run(self) -> None:
        pass


class _SimpleClassifier(nn.Module):
    def __init__(self, in_dim: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        x = input_ids.float() if input_ids.dtype in (torch.long, torch.int32) else input_ids
        if x.dim() == 3:
            x = x.mean(dim=1)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class _ToyDataset(Dataset):
    def __init__(self, size: int = 6, dim: int = 8) -> None:
        self.size = size
        self.dim = dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randn(self.dim),
            "attention_mask": torch.ones(self.dim, dtype=torch.long),
            "labels": torch.tensor(idx % 2, dtype=torch.long),
        }


def test_base_flush_l2_dynamic_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify flush_l2_cache queries L2_cache_size dynamically from device properties."""
    mock_config = MagicMock()
    mock_config.hardware.device = "cuda:0"
    mock_tracker = MagicMock()
    mock_context = MagicMock()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mock_props = MagicMock()
    mock_props.L2_cache_size = 64 * 1024 * 1024  # 64MB
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: mock_props)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev: None)

    allocated: list[int] = []

    def mock_empty(num_elements: int, dtype: torch.dtype, device: torch.device) -> MagicMock:
        allocated.append(num_elements)
        return MagicMock()

    monkeypatch.setattr(torch, "empty", mock_empty)

    bm = _DummyBenchmark(config=mock_config, tracker=mock_tracker, context=mock_context)
    bm.device = torch.device("cuda:0")
    bm.flush_l2_cache()

    assert len(allocated) == 1
    # 64MB + 16MB = 80MB -> 80 * 1024 * 1024 // 4 elements
    expected_elements = (80 * 1024 * 1024) // 4
    assert allocated[0] == expected_elements


def test_evaluation_measure_resources_fragmentation_telemetry() -> None:
    """Verify measure_resources reports memory fragmentation metrics."""
    model = _SimpleClassifier()
    batch = {
        "input_ids": torch.randn(2, 8),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    stats = measure_resources(model, batch, device="cpu", warmup_iters=1, measure_iters=2)
    assert "fragmentation_ratio" in stats
    assert "reserved_mb" in stats
    assert "allocated_mb" in stats
    assert stats["fragmentation_ratio"] >= 0.0


def test_kd_loss_build_teacher_cache_weights_only_warning(tmp_path: Path) -> None:
    """Verify build_teacher_cache emits UserWarning regarding torch.save serialization."""
    model = _SimpleClassifier()
    loader = DataLoader(_ToyDataset(2, 8), batch_size=2)
    cache_file = tmp_path / "teacher_cache.pt"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out_p = build_teacher_cache(model, loader, cache_file, device="cpu")
        assert out_p.exists()
        assert any("torch.save" in str(item.message) for item in w)


def test_finetune_classification_22_parameters(tmp_path: Path) -> None:
    """Verify finetune_classification handles all 22 required parameters."""
    model = _SimpleClassifier()
    loader = DataLoader(_ToyDataset(4, 8), batch_size=2)

    result = finetune_classification(
        model=model,
        train_loader=loader,
        eval_loader=loader,
        epochs=1,
        lr=1e-4,
        device="cpu",
        max_steps=2,
        mixed_precision="no",
        distributed_backend="none",
        tracker=None,
        log_interval=5,
        optimizer_name="adamw",
        seed=42,
        weight_decay=0.01,
        warmup_steps=0,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        eval_steps=None,
        save_steps=None,
        checkpoint_dir=str(tmp_path),
        early_stopping_patience=None,
    )
    assert result["status"] == "success"
    assert result["steps_completed"] == 2


def test_finetune_classification_stiefel_optimizer() -> None:
    """Verify finetune_classification handles stiefel optimizer path cleanly."""
    model = _SimpleClassifier()
    loader = DataLoader(_ToyDataset(2, 8), batch_size=2)
    mock_opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    with patch(
        "neuralops.trainer.optimizers.factory.create_riemannian_stiefel",
        return_value=mock_opt,
    ):
        result = finetune_classification(
            model=model,
            train_loader=loader,
            epochs=1,
            lr=1e-4,
            device="cpu",
            optimizer_name="stiefel",
            max_steps=1,
        )
    assert result["status"] == "success"
    assert result["steps_completed"] == 1


def test_stats_functions_complete() -> None:
    """Verify paired_ttest, bootstrap_ci, and fdr_correction function properly."""
    # Paired t-test
    p_val = paired_ttest([0.9, 0.92, 0.91], [0.85, 0.86, 0.84])
    assert 0.0 <= p_val <= 1.0

    # Bootstrap CI
    low, high = bootstrap_ci([0.8, 0.82, 0.85, 0.9], n_boot=200, ci=0.95, seed=42)
    assert low <= high

    # FDR correction
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = fdr_correction(raw_p, alpha=0.05)
    assert len(adj_p) == len(raw_p)
    assert all(0.0 <= p <= 1.0 for p in adj_p)


def test_storage_path_manager_isolation(tmp_path: Path) -> None:
    """Verify StoragePathManager produces isolated run directory structure and symlinks."""
    ctx = StoragePathManager.resolve_run_dir("qwen_kd_benchmark", run_id="run_001", output_root=tmp_path)
    assert ctx.run_dir.exists()
    assert ctx.artifacts_dir.exists()
    assert ctx.run_id == "run_001"

    StoragePathManager.update_latest_symlink(ctx)
    latest = StoragePathManager.get_latest_run_dir("qwen_kd_benchmark", output_root=tmp_path)
    assert latest is not None
    assert latest.resolve() == ctx.run_dir.resolve()
