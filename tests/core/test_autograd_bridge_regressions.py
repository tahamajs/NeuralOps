#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for custom-kernel autograd bridges."""

from __future__ import annotations

import pytest
import torch

from neuralops.core.cuda_kernels._fallback import dispatch, force_kernel_backend


def test_autograd_bridge_handles_multiple_tensor_outputs() -> None:
    x = torch.tensor([2.0], requires_grad=True)

    def forward(value: torch.Tensor):
        return value.square(), value * 3

    with force_kernel_backend("cuda"):
        first, second = dispatch(forward, None, forward, x, kernel_name="multi_output")
    loss = 2 * first.sum() + 5 * second.sum()

    loss.backward()
    assert torch.equal(x.grad, torch.tensor([23.0]))
