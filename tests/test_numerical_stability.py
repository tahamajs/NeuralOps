# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for safe_softmax and safe_log numerical stability."""
import torch
import torch.nn.functional as F
from neuralops.core.attention_utils import safe_softmax, safe_log


def test_safe_softmax_extreme():
    logits = torch.tensor([1000.0, 1001.0, 1002.0])
    res = safe_softmax(logits, dim=-1)
    assert not torch.isnan(res).any()
    assert torch.allclose(res.sum(), torch.tensor(1.0))


def test_safe_log():
    x = torch.tensor([0.0, 1.0, 10.0])
    res = safe_log(x)
    assert not torch.isneginf(res[0])
    assert not torch.isnan(res).any()
    assert torch.allclose(res[1], torch.tensor(0.0))
