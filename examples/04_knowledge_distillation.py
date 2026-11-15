#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 04: Multi-Level Knowledge Distillation & Post-Surgery Recovery.

Subsystem Demonstrated:
    `neuralops.trainer.distillation.multi_level_kd` components:
    (`LogitDistillationLoss`, `HiddenMSELoss`, `AttentionDistillationLoss`,
     `LinearCKALoss`, `CosineAlignmentLoss`, `GramMatrixLoss`, `MultiLevelKDTrainer`)
    and `neuralops.trainer.schedulers.WSDScheduler`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Multi-level distillation across multiple representation pillars:
       - Pillar 1: Logit KL Divergence (`LogitDistillationLoss`)
       - Pillar 2: Hidden State MSE (`HiddenMSELoss`)
       - Pillar 3: Attention Distribution Parity (`AttentionDistillationLoss`)
       - Pillar 4: Centered Kernel Alignment (`LinearCKALoss`)
       - Pillar 5: Gram Matrix Covariance Alignment (`GramMatrixLoss`)
       - Pillar 6: Angular Directional Alignment (`CosineAlignmentLoss`)
    2. Post-surgery healing on a subquadratic student model (`cosformer`).
    3. Warmup-Stable-Decay (`WSDScheduler`) learning rate scheduling.
    4. Distillation training loop using official `MultiLevelKDTrainer` and standalone losses.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import neuralops
from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    print_benchmark_table,
    section,
    toy_dataloader,
)
from neuralops.trainer.distillation.multi_level_kd import (
    AttentionDistillationLoss,
    CosineAlignmentLoss,
    GramMatrixLoss,
    HiddenMSELoss,
    LinearCKALoss,
    LogitDistillationLoss,
    MultiLevelKDTrainer,
)
from neuralops.trainer.schedulers import WSDScheduler


def run_distillation_workflow(device: str) -> bool:
    """Execute 3-step multi-level distillation training with WSD scheduling using library components."""
    section("Part 1: Initializing Teacher, Student, and Distillation Components")

    teacher = TinyClassifier(seed=42)
    student = TinyClassifier(seed=99)
    # Inject cosformer on CPU before moving to accelerator for fast graph initialization
    neuralops.replace_attention(student, attention="cosformer", copy_weights=False)

    teacher = teacher.to(device)
    student = student.to(device)
    student.train()

    kv("Teacher Architecture", "Dense DistilBERT (2 layers, full quadratic attention)")
    kv("Student Architecture", "Transformed DistilBERT (2 layers, Cosformer linear attention)")
    kv("Distillation Engine", "neuralops.trainer.distillation.multi_level_kd")
    print()

    # Initialize library distillation loss components
    loss_logit_fn = LogitDistillationLoss(temperature=2.0)
    loss_hidden_fn = HiddenMSELoss()
    loss_attn_fn = AttentionDistillationLoss(temperature=2.0)
    loss_cka_fn = LinearCKALoss()
    loss_gram_fn = GramMatrixLoss()
    loss_cos_fn = CosineAlignmentLoss()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = WSDScheduler(optimizer, warmup_steps=1, stable_steps=1, decay_steps=1, min_lr=1e-5)

    data_iter = toy_dataloader(num_batches=3, batch_size=2, seq_len=16, device=device)

    section("Part 2: 3-Step Distillation Loop & Loss Telemetry")
    table_rows: list[list[str]] = []
    losses_history: list[float] = []
    pillar_values_all: list[list[float]] = []

    for step, batch in enumerate(data_iter, start=1):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        # Forward teacher without gradients
        with torch.no_grad():
            t_out = teacher(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            t_logits = t_out.logits
            t_hidden = t_out.hidden_states[-1]

        # Forward student
        s_out = student(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        s_logits = s_out.logits
        s_hidden = s_out.hidden_states[-1]

        # Multi-pillar losses evaluated via official library classes
        p1_logit = loss_logit_fn(s_logits, t_logits)
        p2_mse = loss_hidden_fn(s_hidden, t_hidden)
        p3_attn = loss_attn_fn(s_hidden, t_hidden)
        p4_cka = loss_cka_fn(s_hidden, t_hidden)
        p5_gram = loss_gram_fn(s_hidden, t_hidden)
        p6_cos = loss_cos_fn(s_hidden, t_hidden)

        total_loss = 0.4 * p1_logit + 0.2 * p2_mse + 0.1 * p3_attn + 0.1 * p4_cka + 0.1 * p5_gram + 0.1 * p6_cos

        total_loss.backward()
        optimizer.step()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_val = total_loss.item()
        losses_history.append(loss_val)

        p1_val = p1_logit.item()
        p2_val = p2_mse.item()
        p3_val = p3_attn.item()
        p4_val = p4_cka.item()
        p5_val = p5_gram.item()
        p6_val = p6_cos.item()

        pillar_values_all.append([p1_val, p2_val, p3_val, p4_val, p5_val, p6_val])

        table_rows.append(
            [
                f"Step {step}",
                f"{current_lr:.6f}",
                f"{p1_val:.4f}",
                f"{p2_val:.4f}",
                f"{p3_val:.4f}",
                f"{p4_val:.4f}",
                f"{p5_val:.4f}",
                f"{p6_val:.4f}",
                f"{loss_val:.4f}",
            ]
        )

    print_benchmark_table(
        table_rows,
        headers=("Step", "LR", "P1:Logit", "P2:MSE", "P3:Attn", "P4:CKA", "P5:Gram", "P6:Cos", "Total Loss"),
    )

    all_finite = bool(all(torch.isfinite(torch.tensor(l)).item() for l in losses_history))
    pass_fail("Multi-Level Loss Finiteness", all_finite, f"Loss trajectory: {[round(l, 4) for l in losses_history]}")

    # Check non-zero real computation across all pillars
    nonzero_counts = sum(1 for col in range(6) if any(step_vals[col] > 1e-6 for step_vals in pillar_values_all))
    p_pillars = bool(nonzero_counts == 6)
    pass_fail(
        "Pillar Real Value Verification",
        p_pillars,
        f"{nonzero_counts}/6 distillation pillars computed with nonzero real values",
    )

    # Part 3: Verify MultiLevelKDTrainer execution
    section("Part 3: MultiLevelKDTrainer Integration")
    trainer = MultiLevelKDTrainer(
        teacher=teacher,
        student=student,
        device=device,
        temperature=2.0,
        lambda_kd=1.0,
        lambda_mse=0.5,
        lambda_cka=0.2,
        lambda_cos=0.1,
    )
    batch = next(iter(toy_dataloader(num_batches=1, batch_size=2, seq_len=16, device=device)))
    step_dict = trainer.training_step(batch)
    trainer_loss = step_dict.get("total_loss")
    p_trainer = bool(
        isinstance(trainer_loss, torch.Tensor) and torch.isfinite(trainer_loss).item() and trainer_loss.item() > 0
    )
    pass_fail(
        "MultiLevelKDTrainer Step",
        p_trainer,
        f"Trainer total loss={trainer_loss.item():.4f}" if p_trainer else "Invalid trainer step",
    )

    p_wsd = bool(current_lr > 0)
    pass_fail("WSD Learning Rate Scheduling", p_wsd, "Successfully stepped WSD scheduler")

    return bool(all_finite and p_pillars and p_trainer and p_wsd)


def main() -> int:
    section("NeuralOps Example 04: Multi-Level Knowledge Distillation")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    success = bool(run_distillation_workflow(backend))
    print()
    pass_fail(
        "Overall Example 04 Status",
        success,
        "All library distillation pillars and trainer checks passed"
        if success
        else "One or more distillation checks failed",
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
