#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for CUDAGraphRunner."""
import torch
from neuralops.core.cuda_graph import CUDAGraphRunner


def test_cuda_graph_runner_cpu_fallback():
    def add_one(x):
        return x + 1.0

    runner = CUDAGraphRunner(add_one)
    x = torch.tensor([1.0, 2.0, 3.0])
    runner.capture(x)
    out = runner.replay(x)
    assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0]))
