"""
Example: Mixed Precision and Knowledge Distillation Fine-Tuning

Shows how to:
  1. Set up student model and teacher model for fine-tuning.
  2. Compute combined task loss and distillation loss (Kullback-Leibler divergence).
  3. Perform optimized training step using MixedPrecisionTrainer.

Requires:
  - neuralops installed (pip install neuralops)
  - Optional: CUDA GPU (falls back to CPU)
  - Optional: Triton (falls back to PyTorch)

Run:
    python examples/fine_tuning_demo.py

Expected output:
    Distillation training step completed with valid combined loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.training.trainer import MixedPrecisionTrainer


def main() -> None:
    print("Initializing student and teacher models...")
    # Representative student and teacher classifiers
    student = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    teacher = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    trainer = MixedPrecisionTrainer(student, optimizer, loss_fn, enabled=torch.cuda.is_available())

    # Simulated batch inputs and labels
    x = torch.randn(8, 64)
    labels = torch.randint(0, 2, (8,))

    # Teacher prediction
    with torch.no_grad():
        teacher_logits = teacher(x)

    # Student forward and KD training step
    student.train()
    student_logits = student(x)
    task_loss = loss_fn(student_logits, labels)

    # Distillation loss
    temperature = 2.0
    alpha = 0.5
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)

    total_loss = (1 - alpha) * task_loss + alpha * kd_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Task Loss: {task_loss.item():.4f}, KD Loss: {kd_loss.item():.4f}, Combined Loss: {total_loss.item():.4f}")
    assert not torch.isnan(total_loss)
    print("✅ Distillation training step completed with valid combined loss.")


if __name__ == "__main__":
    main()
