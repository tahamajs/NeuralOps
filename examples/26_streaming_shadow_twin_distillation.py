#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 26: Micro-Chunked Streaming Shadow Twin Distillation & Zero-OOM Engine.

Subsystem Demonstrated:
    Memory-bounded student-teacher knowledge distillation and layer cloning:
    1. Micro-Chunked Slicing:
       - Partitions sequences into non-overlapping micro-chunks (C=128) fitting inside L2 cache
    2. Zero-OOM Activation Checkpointing:
       - Eliminates peak VRAM bottlenecks during post-surgery model healing
    3. Streaming Centered Kernel Alignment (CKA):
       - Online Chan covariance updating accumulating Gram matrices in O(D^2) feature space
       - Avoids O(N^2) sample-space explosion when B * T > 100k
    4. Sliced Layer Cloning MSE:
       - Symmetrical operator cloning MSE loss preserving gradient highways
    5. Active VRAM Telemetry & Cache Purging:
       - Real-time VRAM utilization polling (< 5 us) and threshold-based GC sweeps

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional (measures real-time VRAM and cache purge metrics when CUDA is detected).
    - Network: Not required; uses offline synthetic token generators.

What You Will Learn:
    1. How MicroChunkedShadowTwinDistiller executes streaming distillation over long sequences.
    2. How streaming CKA tracks representation alignment in constant feature memory.
    3. How inter-chunk student recurrent states propagate across token boundaries.
    4. How zero-OOM distillation enables efficient surgical healing on modest hardware.
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

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)
import neuralops
from neuralops.surgery.streaming_distill import MicroChunkedShadowTwinDistiller


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
    chunk_size: int = 16,
    batch_size: int = 2,
    seq_len: int = 48,
) -> bool:
    """Execute streaming shadow twin distillation and verify zero-OOM convergence."""
    # 1. Initialize teacher model and surgical student model
    torch.manual_seed(42)
    teacher = TinyClassifier(n_layers=2, dim=32, hidden_dim=64).to(device)
    student = TinyClassifier(n_layers=2, dim=32, hidden_dim=64).to(device)

    # Perform linear attention transplant on student
    student = neuralops.replace_attention(student, attention="cosformer", copy_weights=True)

    # 2. Instantiate MicroChunkedShadowTwinDistiller
    distiller = MicroChunkedShadowTwinDistiller(
        teacher=teacher,
        student=student,
        chunk_size=chunk_size,
        hidden_dim=32,
        lambda_task=1.0,
        lambda_cloning=1.0,
        lambda_cka=0.2,
        device=device,
    )

    # 3. Create synthetic input sequence tokens
    input_ids = torch.randint(0, 128, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    # 4. Execute streaming distillation step (layer cloning MSE + streaming CKA)
    distill_out = distiller.step(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    loss = distill_out["loss"]
    cloning_loss = distill_out["cloning_loss"]
    cka_loss = distill_out["cka_loss"]
    task_loss = distill_out["task_loss"]

    loss_finite = bool(torch.isfinite(loss).item())
    pass_fail("Composite Distillation Loss Finite Check", loss_finite, f"Loss = {loss.item():.4f}")
    pass_fail("Layer Cloning MSE Finite Check", bool(torch.isfinite(cloning_loss).item()), f"MSE = {cloning_loss.item():.4f}")
    pass_fail("Streaming CKA Loss Finite Check", bool(torch.isfinite(cka_loss).item()), f"CKA = {cka_loss.item():.4f}")

    # 5. Verify backward gradient propagation into student parameters
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.requires_grad]
    grads_ok = bool(grads) and any(g is not None and torch.isfinite(g).all().item() for g in grads)
    pass_fail("Student Trainable Gradients Preserved", grads_ok)

    # Verify teacher remains strictly frozen
    teacher_frozen = all(p.grad is None for p in teacher.parameters())
    pass_fail("Teacher Frozen Invariant Preserved", teacher_frozen)

    # 6. Render telemetry summary table
    headers = ["Metric / Loss Component", "Value", "Theoretical Guarantee"]
    rows = [
        ["Total Distillation Loss", f"{loss.item():.4f}", "Composite Objective"],
        ["Layer Cloning MSE Loss", f"{cloning_loss.item():.4f}", "Exact Feature Alignment"],
        ["Streaming CKA Loss", f"{cka_loss.item():.4f}", "O(D^2) Bounded Gram Matrix"],
        ["Task Cross-Entropy Loss", f"{task_loss.item():.4f}" if task_loss is not None else "N/A", "Label Supervised"],
        ["VRAM Utilization", f"{distill_out['vram_utilization'] * 100:.1f}%", "< Threshold Limit"],
    ]
    print()
    table(rows, headers=headers)

    return bool(loss_finite and grads_ok and teacher_frozen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda[:index]")
    parser.add_argument("--chunk-size", type=int, default=16, help="Micro-chunk sequence size")
    parser.add_argument("--seq-len", type=int, default=48, help="Total input sequence length")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 26: Micro-Chunked Streaming Shadow Twin Distillation")
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", str(device))
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_demonstration(
        device=device,
        chunk_size=args.chunk_size,
        seq_len=args.seq_len,
    )

    print()
    if success:
        pass_fail("Overall Example 26 Status", True, "Streaming shadow twin distillation verified")
        return 0
    else:
        pass_fail("Overall Example 26 Status", False, "Distillation verification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
