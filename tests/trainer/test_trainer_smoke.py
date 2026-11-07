#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke tests for neuralops.trainer subsystem."""

import torch
import torch.nn as nn
from neuralops.trainer.schedulers import get_cosine_schedule_with_warmup


def test_trainer_scheduler_smoke():
    """Verify trainer scheduler utility computes valid learning rate steps."""
    model = nn.Linear(8, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    assert scheduler is not None

    # Step through warmup
    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    next_lr = optimizer.param_groups[0]["lr"]
    assert next_lr >= 0.0
