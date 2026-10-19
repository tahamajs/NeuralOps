# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for softcap function."""
import torch
import pytest
from neuralops.core.attention_utils import softcap


def test_softcap_bounds():
    scores = torch.randn(10, 10) * 1000.0
    out = softcap(scores, cap=50.0)
    assert (out.abs() <= 50.0).all()


def test_softcap_monotonicity():
    x = torch.linspace(-100, 100, 200)
    out = softcap(x, cap=30.0)
    diff = out[1:] - out[:-1]
    assert (diff >= 0).all()


def test_softcap_gradients():
    x = torch.randn(5, 5, requires_grad=True)
    out = softcap(x, cap=20.0)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
