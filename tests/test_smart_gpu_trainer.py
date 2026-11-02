#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Smart GPU-Maximizing Training Engine (AutoMaxGPUTrainer).
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralops.trainer.smart_gpu_trainer import (
    AsyncDataPrefetcher,
    AutoMaxGPUTrainer,
    GPUMemoryManager,
)
from neuralops.trainer.trainer_factory import get_trainer


class SimpleDataset(Dataset):
    """Synthetic dataset for trainer verification."""

    def __init__(self, size: int = 64, dim: int = 16):
        self.size = size
        self.dim = dim
        self.x = torch.randn(size, dim)
        self.y = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.x[idx], "labels": self.y[idx]}


class SimpleModel(nn.Module):
    """Simple linear classification model for trainer tests."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
        logits = self.fc(input_ids)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return type("Output", (), {"logits": logits, "loss": loss})()


def test_gpu_memory_manager_cpu():
    """Verify GPUMemoryManager handles CPU/non-CUDA gracefully."""
    device = torch.device("cpu")
    info = GPUMemoryManager.get_device_info(device)
    assert info["is_cuda"] is False
    assert info["device_name"] == "CPU/MPS"

    ratio = GPUMemoryManager.get_memory_allocated_ratio(device)
    assert ratio == 0.0

    # Test empty cache call does not crash
    GPUMemoryManager.empty_cache(device)


def test_async_data_prefetcher():
    """Verify prefetcher correctly yields batches and stages to device."""
    ds = SimpleDataset(size=16, dim=8)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    device = torch.device("cpu")

    prefetcher = AsyncDataPrefetcher(loader, device)
    batches = list(prefetcher)
    assert len(batches) == 4
    for b in batches:
        assert "input_ids" in b
        assert "labels" in b
        assert b["input_ids"].device == device


def test_automax_gpu_trainer_training_loop():
    """Verify AutoMaxGPUTrainer executes training loop, evaluation, and logging."""
    train_ds = SimpleDataset(size=32, dim=8)
    eval_ds = SimpleDataset(size=16, dim=8)
    model = SimpleModel(dim=8)

    trainer = AutoMaxGPUTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        device="cpu",
        epochs=1,
        initial_micro_batch_size=4,
        max_micro_batch_size=8,
        gradient_accumulation_steps=2,
        use_amp=False,
        max_grad_norm=1.0,
        logging_steps=1,
    )

    metrics = trainer.train()
    assert "epochs_completed" in metrics
    assert metrics["epochs_completed"] == 1
    assert "final_loss" in metrics
    assert "throughput_samples_per_sec" in metrics

    eval_metrics = trainer.evaluate()
    assert "eval_loss" in eval_metrics
    assert eval_metrics["eval_loss"] >= 0.0


def test_automax_gpu_trainer_oom_recovery():
    """Verify that OOM error triggers micro-batch halving, grad accum doubling, and retry."""
    train_ds = SimpleDataset(size=32, dim=8)
    model = SimpleModel(dim=8)

    trainer = AutoMaxGPUTrainer(
        model=model,
        train_dataset=train_ds,
        device="cpu",
        epochs=1,
        initial_micro_batch_size=8,
        gradient_accumulation_steps=1,
        use_amp=False,
    )

    initial_batch = trainer.micro_batch_size
    initial_accum = trainer.grad_accum_steps

    # Mock an OOM condition on first call, then pass on subsequent
    call_count = 0
    orig_forward = model.forward

    def mock_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM in test")
        return orig_forward(*args, **kwargs)

    model.forward = mock_forward

    metrics = trainer.train()
    assert trainer.micro_batch_size < initial_batch
    assert trainer.grad_accum_steps > initial_accum
    assert metrics["epochs_completed"] == 1


def test_trainer_factory_registration():
    """Verify AutoMaxGPUTrainer is retrievable via factory registry."""
    cls = get_trainer("automax_gpu")
    assert cls is AutoMaxGPUTrainer
