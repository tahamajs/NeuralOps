# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for selective gradient checkpointing."""
import torch
import torch.nn as nn
from neuralops.training.checkpointing import checkpoint_selective, SelectiveCheckpointModule


def test_checkpoint_selective():
    lin = nn.Linear(8, 8)
    x = torch.randn(2, 8, requires_grad=True)

    out_cp = checkpoint_selective(lin, x, should_checkpoint=True)
    out_cp.sum().backward()
    assert x.grad is not None

    x2 = torch.randn(2, 8, requires_grad=True)
    out_no = checkpoint_selective(lin, x2, should_checkpoint=False)
    out_no.sum().backward()
    assert x2.grad is not None


def test_selective_checkpoint_module():
    lin = nn.Linear(8, 8)
    wrapper = SelectiveCheckpointModule(lin, min_seqlen_threshold=16)

    x_short = torch.randn(2, 8, 8, requires_grad=True)
    out1 = wrapper(x_short)
    assert out1.shape == (2, 8, 8)

    x_long = torch.randn(2, 32, 8, requires_grad=True)
    out2 = wrapper(x_long)
    out2.sum().backward()
    assert x_long.grad is not None
