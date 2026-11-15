#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and unit regression tests for examples/28_smart_trainer_automax_vram.py.
"""

from __future__ import annotations

import subprocess
import sys
import torch

from examples._shared import TinyClassifier
from neuralops.trainer.smart_gpu_trainer import AutoMaxGPUTrainer, GPUMemoryManager


def test_28_runs_clean():
    """Verify example 28 executes cleanly via CLI subprocess with exit code 0."""
    proc = subprocess.run(
        [
            sys.executable,
            "examples/28_smart_trainer_automax_vram.py",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--global-batch-size",
            "8",
            "--micro-batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 28 failed with stderr:\n{proc.stderr}"
    assert "[PASS] Overall Example 28 Status" in proc.stdout


def test_28_automax_memory_manager_cpu_fallback():
    """Verify GPUMemoryManager handles CPU / non-CUDA devices safely without raising."""
    cpu_device = torch.device("cpu")
    info = GPUMemoryManager.get_device_info(cpu_device)
    assert info["is_cuda"] is False
    assert info["device_name"] == "CPU/MPS"
    assert info["total_memory_mb"] == 0.0

    ratio = GPUMemoryManager.get_memory_allocated_ratio(cpu_device)
    assert ratio == 0.0

    # empty_cache should succeed without exception
    GPUMemoryManager.empty_cache(cpu_device)
    GPUMemoryManager.clear_cache()


def test_28_automax_trainer_batch_invariant():
    """Verify AutoMax trainer calibrates accumulation steps to satisfy global batch size."""
    # Synthetic dataset
    class MiniDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 50, (16,), dtype=torch.long),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.tensor(0, dtype=torch.long),
            }

    model = TinyClassifier(n_layers=1, dim=16, hidden_dim=32, num_labels=2)
    config = {
        "global_batch_size": 16,
        "micro_batch_size": 4,
        "epochs": 1,
        "use_amp": False,
    }

    trainer = AutoMaxGPUTrainer(
        model=model,
        train_dataset=MiniDataset(),
        config=config,
        device=torch.device("cpu"),
    )

    assert trainer.micro_batch_size == 4
    assert trainer.grad_accum_steps == 4
    assert trainer.micro_batch_size * trainer.grad_accum_steps == 16
