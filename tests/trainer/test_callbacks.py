#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and regression tests for Trainer Callbacks.

Verifies on_log, on_evaluate, on_train_end hooks with HuggingFace Trainer,
accurate parameter counting in SVDAttentionCallback, memory reporting,
gradient norm tracking, and preemption checkpoint saving.
"""

import os
import tempfile
import time
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from neuralops.trainer.callbacks import (
    GradientNormCallback,
    MemoryCallback,
    PreemptionCallback,
    SVDAttentionCallback,
    TimeCallback,
)


class DummySVDLinearLayer(nn.Module):
    """Dummy SVD layer for compression testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


class NestedBlock(nn.Module):
    """Hierarchical block with nested layers."""

    def __init__(self) -> None:
        super().__init__()
        self.svd = DummySVDLinearLayer(10, 10)  # 100 params
        self.fc = nn.Linear(10, 10, bias=True)  # 110 params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.svd(x))


class MultiLevelModel(nn.Module):
    """Multi-level nested module tree to test parameter counting."""

    def __init__(self) -> None:
        super().__init__()
        self.b1 = NestedBlock()  # 210 params (100 svd + 110 fc)
        self.b2 = NestedBlock()  # 210 params (100 svd + 110 fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b2(self.b1(x))


# =====================================================================
# SVDAttentionCallback Tests
# =====================================================================


def test_svd_attention_callback_accurate_parameter_counting() -> None:
    """Verify SVDAttentionCallback accurately counts total parameters without hierarchy multiplication."""
    model = MultiLevelModel()
    true_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert true_total == 420  # 100 + 110 + 100 + 110

    cb = SVDAttentionCallback(log_attention_stats=True)
    logs: Dict[str, Any] = {}
    state = TrainerState()
    control = TrainerControl()
    args = TrainingArguments(output_dir="/tmp/test_svd_cb")

    cb.on_log(args, state, control, logs=logs, model=model)

    assert "svd_layers" in logs
    assert logs["svd_layers"] == 2
    assert "compression_ratio" in logs
    # 2 SVD layers * 100 params = 200 params. 200 / 420 = 0.47619...
    expected_ratio = 200.0 / 420.0
    assert abs(logs["compression_ratio"] - expected_ratio) < 1e-4


def test_svd_attention_callback_hooks_on_evaluate_and_train_end() -> None:
    """Verify on_evaluate and on_train_end hooks in SVDAttentionCallback."""
    model = MultiLevelModel()
    cb = SVDAttentionCallback(log_attention_stats=True)
    metrics: Dict[str, Any] = {}
    state = TrainerState()
    control = TrainerControl()
    args = TrainingArguments(output_dir="/tmp/test_svd_cb")

    cb.on_evaluate(args, state, control, metrics=metrics, model=model)
    assert "svd_layers" in metrics
    assert "compression_ratio" in metrics
    assert metrics["svd_layers"] == 2

    cb.on_train_end(args, state, control, model=model)
    assert hasattr(cb, "last_svd_layers")
    assert cb.last_svd_layers == 2


# =====================================================================
# MemoryCallback Tests
# =====================================================================


def test_memory_callback_on_log_and_on_evaluate() -> None:
    """Verify MemoryCallback records memory in on_log and on_evaluate."""
    cb = MemoryCallback()
    logs: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    state = TrainerState()
    control = TrainerControl()
    args = TrainingArguments(output_dir="/tmp/test_mem_cb")

    cb.on_log(args, state, control, logs=logs)
    cb.on_evaluate(args, state, control, metrics=metrics)

    if torch.cuda.is_available():
        assert "gpu_memory_allocated" in logs
        assert "gpu_memory_reserved" in logs
        assert "gpu_peak_memory" in logs
        assert "gpu_memory_allocated" in metrics
        assert "gpu_memory_reserved" in metrics
        assert "gpu_peak_memory" in metrics
    elif torch.backends.mps.is_available():
        assert "mps_memory" in logs
        assert "mps_memory" in metrics


def test_memory_callback_on_train_end() -> None:
    """Verify MemoryCallback on_train_end handles completion cleanly."""
    cb = MemoryCallback()
    state = TrainerState()
    control = TrainerControl()
    args = TrainingArguments(output_dir="/tmp/test_mem_cb")
    cb.on_train_end(args, state, control)


# =====================================================================
# TimeCallback Tests
# =====================================================================


def test_time_callback_on_log_on_evaluate_on_train_end() -> None:
    """Verify TimeCallback measures elapsed and total time across all hooks."""
    cb = TimeCallback()
    args = TrainingArguments(output_dir="/tmp/test_time_cb")
    state = TrainerState()
    control = TrainerControl()

    cb.on_train_begin(args, state, control)
    time.sleep(0.02)

    logs: Dict[str, Any] = {}
    cb.on_log(args, state, control, logs=logs)
    assert "elapsed_time" in logs
    assert logs["elapsed_time"] > 0.01

    metrics: Dict[str, Any] = {}
    cb.on_evaluate(args, state, control, metrics=metrics)
    assert "elapsed_time" in metrics
    assert metrics["elapsed_time"] >= logs["elapsed_time"]

    cb.on_train_end(args, state, control)
    assert cb.total_time is not None
    assert cb.total_time >= metrics["elapsed_time"]


# =====================================================================
# GradientNormCallback Tests
# =====================================================================


def test_gradient_norm_callback_records_norm_in_logs_and_metrics() -> None:
    """Verify GradientNormCallback computes correct norm and exposes it via on_log/on_evaluate."""
    model = nn.Linear(5, 2)
    cb = GradientNormCallback(log_every_n_steps=1)
    args = TrainingArguments(output_dir="/tmp/test_gn_cb")
    state = TrainerState()
    state.global_step = 1
    control = TrainerControl()

    x = torch.randn(2, 5)
    loss = model(x).sum()
    loss.backward()

    # Calculate expected norm
    expected_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            expected_norm += p.grad.norm(2).item() ** 2
    expected_norm = expected_norm**0.5

    cb.on_step_end(args, state, control, model=model)
    assert cb.last_grad_norm is not None
    assert abs(cb.last_grad_norm - expected_norm) < 1e-4

    logs: Dict[str, Any] = {}
    cb.on_log(args, state, control, logs=logs)
    assert "grad_norm" in logs
    assert abs(logs["grad_norm"] - expected_norm) < 1e-4

    metrics: Dict[str, Any] = {}
    cb.on_evaluate(args, state, control, metrics=metrics)
    assert "grad_norm" in metrics
    assert abs(metrics["grad_norm"] - expected_norm) < 1e-4


# =====================================================================
# PreemptionCallback Tests
# =====================================================================


def test_preemption_callback_triggers_stop_and_save() -> None:
    """Verify PreemptionCallback sets both should_training_stop and should_save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        job_id = "test_preempt_123"
        flag_file = os.path.join(tmpdir, job_id)

        cb = PreemptionCallback(preempt_flag_dir=tmpdir, job_id=job_id)
        control = TrainerControl()
        state = TrainerState()
        state.global_step = 10
        args = TrainingArguments(output_dir=tmpdir)

        # Before flag file exists
        cb.on_step_end(args, state, control)
        assert not control.should_training_stop
        assert not control.should_save

        # Create flag file
        with open(flag_file, "w") as f:
            f.write("stop")

        cb.on_step_end(args, state, control)
        assert control.should_training_stop
        assert control.should_save


def test_preemption_callback_environment_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PreemptionCallback detects SLURM_JOB_ID when job_id is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("SLURM_JOB_ID", "slurm_999")
        flag_file = os.path.join(tmpdir, "slurm_999")
        with open(flag_file, "w") as f:
            f.write("1")

        cb = PreemptionCallback(preempt_flag_dir=tmpdir, job_id=None)
        control = TrainerControl()
        state = TrainerState()
        args = TrainingArguments(output_dir=tmpdir)

        cb.on_step_end(args, state, control)
        assert control.should_training_stop
        assert control.should_save
