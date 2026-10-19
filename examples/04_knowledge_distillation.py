#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run one tiny teacher-student knowledge-distillation step."""

import torch
import torch.nn.functional as F

import neuralops
from _tiny import tiny_classifier, tiny_inputs


def main() -> None:
    teacher = tiny_classifier().eval()
    student = neuralops.replace_attention(tiny_classifier(), "cosformer", copy_weights=False)
    inputs = tiny_inputs()
    with torch.no_grad():
        teacher_logits = teacher(**inputs).logits
    student_logits = student(**inputs).logits
    temperature = 2.0
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    kd_loss.backward()
    assert torch.isfinite(kd_loss)
    print(f"KD step: teacher/student logits={tuple(student_logits.shape)}; loss={kd_loss.item():.5f}")


if __name__ == "__main__":
    main()
