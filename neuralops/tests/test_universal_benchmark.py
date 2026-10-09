"""Unit and integration tests for the universal benchmark system."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import torch

from neuralops.benchmarks.data.loader import (
    DATASET_LOADER_REGISTRY,
    auto_detect_columns,
    detect_dataset_task,
    load_universal_dataset,
    make_universal_dataloader,
)
from neuralops.benchmarks.metrics.evaluator import compute_task_metrics
from neuralops.benchmarks.models.loader import inspect_model_metadata
from neuralops.benchmarks.tasks.adapter import (
    TASK_ADAPTER_REGISTRY,
    TaskAdapter,
    UniversalClassificationAdapter,
    select_universal_task_adapter,
)


def test_task_adapter_registry() -> None:
    """Verify that expected task types are registered."""
    expected_tasks = [
        "classification",
        "question_answering",
        "summarization",
        "translation",
        "instruction_tuning",
        "language_modeling",
        "regression",
        "multiple_choice",
    ]
    for task in expected_tasks:
        assert task in TASK_ADAPTER_REGISTRY, f"Task {task} missing from TASK_ADAPTER_REGISTRY"

    # Test selector
    adapter_cls = select_universal_task_adapter("summarization")
    assert issubclass(adapter_cls, TaskAdapter)

    # Test fallback
    fallback_cls = select_universal_task_adapter("unknown_arbitrary_task")
    assert fallback_cls == UniversalClassificationAdapter


def test_detect_dataset_task_and_columns() -> None:
    """Test column detection and task heuristics."""
    # Classification headers
    cols = ["sentence", "label"]
    t, p, l = auto_detect_columns(cols)
    assert t == "sentence"
    assert p is None
    assert l == "label"
    assert detect_dataset_task(cols) == "classification"

    # Question Answering headers
    qa_cols = ["question", "context", "answer"]
    t, p, l = auto_detect_columns(qa_cols)
    assert t == "question"
    assert p == "context"
    assert l == "answer"
    assert detect_dataset_task(qa_cols) == "question_answering"

    # Summarization headers
    sum_cols = ["document", "summary"]
    assert detect_dataset_task(sum_cols) == "summarization"


def test_dataset_loader_registry() -> None:
    """Verify file format loaders are registered."""
    assert "csv" in DATASET_LOADER_REGISTRY
    assert "json" in DATASET_LOADER_REGISTRY
    assert "parquet" in DATASET_LOADER_REGISTRY


class DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    def __call__(self, texts, texts2=None, **kwargs):
        batch_size = len(texts)
        seq_len = kwargs.get("max_length", 16)
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


def test_load_local_csv_dataset(tmp_path: Path) -> None:
    """Test loading and tokenizing a local CSV file."""
    csv_file = tmp_path / "dataset.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "sentiment"])
        for i in range(10):
            writer.writerow([f"Sample {i}", i % 2])

    tokenizer = DummyTokenizer()
    dataset, metadata = load_universal_dataset(
        dataset_name_or_path=str(csv_file),
        tokenizer=tokenizer,
        task="classification",
        split="train",
        max_length=16,
    )

    assert len(dataset) == 10
    assert metadata.num_train == 10
    assert metadata.task == "classification"

    loader = make_universal_dataloader(dataset, batch_size=2)
    batch = next(iter(loader))
    assert len(batch) >= 3  # input_ids, attention_mask, labels
    assert batch[0].shape == (2, 16)


def test_compute_task_metrics() -> None:
    """Test universal metrics calculation for classification and regression."""
    # Classification metrics
    preds = [0, 1, 1, 0, 1]
    targets = [0, 1, 0, 0, 1]
    metrics = compute_task_metrics("classification", preds, targets)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] == 0.8

    # Regression metrics
    reg_preds = [1.0, 2.0, 3.0]
    reg_targets = [1.1, 2.0, 2.9]
    reg_metrics = compute_task_metrics("regression", reg_preds, reg_targets)
    assert "mse" in reg_metrics
    assert reg_metrics["mse"] >= 0.0


def test_train_with_training_types() -> None:
    """Verify training regimes: none, recovery, kd, and finetune."""
    from neuralops.benchmarks.core.training import train_with_training_type
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 2)
        def forward(self, input_ids, attention_mask=None, labels=None):
            x = input_ids.float()
            logits = self.linear(x)
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            from collections import namedtuple
            Output = namedtuple("Output", ["loss", "logits"])
            return Output(loss=loss, logits=logits)

    student = SimpleModel()
    teacher = SimpleModel()
    inputs = torch.randn(4, 16)
    labels = torch.randint(0, 2, (4,))
    loader = [(inputs, inputs, labels)]

    # Test training_type="none"
    res_none = train_with_training_type(student, loader, training_type="none")
    assert res_none["status"] == "skipped"

    # Test training_type="recovery"
    res_rec = train_with_training_type(student, loader, training_type="recovery", recovery_steps=2)
    assert res_rec["status"] == "success"

    # Test training_type="kd"
    res_kd = train_with_training_type(
        student, loader, training_type="kd", teacher=teacher, epochs=1, max_steps=2
    )
    assert res_kd["training_type"] == "kd"
    assert res_kd["status"] == "success"

