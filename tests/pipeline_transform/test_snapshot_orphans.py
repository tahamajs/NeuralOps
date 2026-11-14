#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for snapshot restoration with orphaned parameters and submodules.
Ensures parameters and modules added during mutation are cleanly removed on restore.
"""

import torch
import torch.nn as nn

from neuralops.pipeline_transform.snapshot import capture, restore


def test_restore_removes_orphaned_parameters_and_submodules():
    """Verify that parameters and submodules added during mutation are removed on restore."""
    model = nn.Sequential(nn.Linear(16, 16))
    snapshot = capture(model)

    # Mutate by adding new parameter and new submodule
    model.orphan_param = nn.Parameter(torch.ones(16))
    model.orphan_module = nn.Linear(8, 8)
    model.register_buffer("orphan_buffer", torch.zeros(4))

    assert hasattr(model, "orphan_param")
    assert hasattr(model, "orphan_module")
    assert hasattr(model, "orphan_buffer")

    # Restore
    restore(model, snapshot)

    assert not hasattr(model, "orphan_param")
    assert not hasattr(model, "orphan_module")
    assert not hasattr(model, "orphan_buffer")
