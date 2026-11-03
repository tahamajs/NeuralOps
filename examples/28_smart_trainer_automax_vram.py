#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 28: Smart AutoMax VRAM GPU Trainer & Dynamic Batch Scheduling.

Subsystem Demonstrated:
    Adaptive hardware saturation search and resilient deep learning training:
    1. Adaptive VRAM Saturation Search:
       - Automatically probes micro-batch sizes to maximize hardware throughput
       - Bounds peak VRAM under target utilization threshold (default 90%)
    2. Dynamic Gradient Accumulation Scaling:
       - Recalibrates accumulation steps to lock effective global batch size invariant:
         effective_batch = micro_batch_size * grad_accum_steps
    3. Asynchronous Prefetching & Double-Buffered CUDA Streams:
       - Overlaps host-to-device memory copies with kernel execution
    4. Transparent Activation Checkpointing:
       - Automatically instruments transformer layers to reduce intermediate activation memory
    5. Real-Time OOM Catching & Non-Destructive Replay:
       - Intercepts OutOfMemory exceptions, clears device cache, halves micro-batch size,
         and resumes training without dropping parameter gradients

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution with synthetic PyTorch datasets.
    - GPU: Optional (dynamically probes physical VRAM allocation and AMP when CUDA is detected).
    - Network: Not required; uses offline synthetic dataset generators.

What You Will Learn:
    1. How AutoMaxGPUTrainer discovers optimal micro-batch sizes automatically.
    2. How gradient accumulation steps scale inversely with micro-batch size.
    3. How memory manager profiles device allocation and prevents OOM crashes.
    4. How to train models with guaranteed batch size invariants across CPU and GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
)
from neuralops.trainer.smart_gpu_trainer import AutoMaxGPUTrainer, GPUMemoryManager


class SyntheticClassificationDataset(Dataset):
    """Deterministic synthetic dataset generating token IDs and labels for sequence classification."""

    def __init__(self, num_samples: int = 32, seq_len: int = 32, num_labels: int = 2) -> None:
        torch.manual_seed(42)
        self.input_ids = torch.randint(0, 128, (num_samples, seq_len), dtype=torch.long)
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
        self.labels = torch.randint(0, num_labels, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict fail-closed policy on CUDA availability."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback.")
    if requested == "auto":
        pref = get_backend_preference()
        return torch.device(pref)
    return torch.device(requested)


def run_demonstration(
    device: torch.device,
    target_global_batch_size: int = 16,
    initial_micro_batch_size: int = 4,
    epochs: int = 2,
) -> bool:
    """Execute AutoMaxGPUTrainer configuration, batch calibration, and multi-epoch training."""
    # 1. Query device memory metrics through GPUMemoryManager
    dev_info = GPUMemoryManager.get_device_info(device)
    kv("Device Type", dev_info["device_name"])
    kv("Is CUDA Enabled", dev_info["is_cuda"])
    if dev_info["is_cuda"]:
        kv("Total Memory", f"{dev_info['total_memory_mb']:.1f} MB")
        kv("Available Memory", f"{dev_info['free_mb']:.1f} MB")
    print()

    # 2. Initialize synthetic dataset and tiny classifier
    train_dataset = SyntheticClassificationDataset(num_samples=32, seq_len=32)
    model = TinyClassifier(n_layers=2, dim=32, hidden_dim=64, num_labels=2)

    # 3. Configure AutoMaxGPUTrainer
    trainer_config = {
        "global_batch_size": target_global_batch_size,
        "micro_batch_size": initial_micro_batch_size,
        "epochs": epochs,
        "target_vram_ratio": 0.85,
        "use_amp": device.type in ("cuda", "mps"),
        "logging_steps": 2,
    }

    trainer = AutoMaxGPUTrainer(
        model=model,
        train_dataset=train_dataset,
        config=trainer_config,
        device=device,
    )

    micro_batch = trainer.micro_batch_size
    accum_steps = trainer.grad_accum_steps
    effective_batch = micro_batch * accum_steps

    kv("Configured Micro-Batch Size", micro_batch)
    kv("Calculated Accumulation Steps", accum_steps)
    kv("Effective Global Batch Size", effective_batch)

    # Verify global batch size invariant
    batch_invariant_ok = effective_batch >= target_global_batch_size or micro_batch == target_global_batch_size
    pass_fail("Global Batch Size Invariant Check", batch_invariant_ok, f"{effective_batch} >= {target_global_batch_size}")

    # 4. Execute multi-epoch training run
    train_results = trainer.train()
    history = train_results.get("history", [])

    train_completed = len(history) == epochs
    losses = [h["train_loss"] for h in history]
    losses_finite = bool(all(torch.isfinite(torch.tensor(l)).item() for l in losses))

    pass_fail("Training Epochs Completed", train_completed, f"{len(history)}/{epochs}")
    pass_fail("Loss Trajectory Finite Check", losses_finite, f"Final Loss = {losses[-1]:.4f}")

    # 5. Render training summary table
    headers = ["Epoch", "Train Loss", "Micro-Batch Size", "Accumulation Steps", "Effective Batch Size"]
    rows = []
    for h in history:
        rows.append([
            f"Epoch {h['epoch']}",
            f"{h['train_loss']:.4f}",
            str(micro_batch),
            str(accum_steps),
            str(effective_batch),
        ])

    print()
    table(rows, headers=headers)

    return bool(batch_invariant_ok and train_completed and losses_finite)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda[:index]")
    parser.add_argument("--global-batch-size", type=int, default=16, help="Target global effective batch size")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Starting micro-batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 28: Smart AutoMax VRAM GPU Trainer")
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", str(device))
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_demonstration(
        device=device,
        target_global_batch_size=args.global_batch_size,
        initial_micro_batch_size=args.micro_batch_size,
        epochs=args.epochs,
    )

    print()
    if success:
        pass_fail("Overall Example 28 Status", True, "AutoMax GPU training completed successfully")
        return 0
    else:
        pass_fail("Overall Example 28 Status", False, "AutoMax GPU training verification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
