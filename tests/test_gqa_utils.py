#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for repeat_kv_optimized."""
import torch
from neuralops.core.utils.gqa import repeat_kv, repeat_kv_optimized


def test_repeat_kv_optimized():
    x = torch.randn(2, 4, 16, 32)
    rep = 2
    out_orig = repeat_kv(x, rep)
    out_opt = repeat_kv_optimized(x, rep)
    assert torch.allclose(out_orig, out_opt)
    assert out_opt.shape == (2, 8, 16, 32)


def test_repeat_kv_optimized_identity():
    x = torch.randn(2, 4, 16, 32)
    out_opt = repeat_kv_optimized(x, 1)
    assert out_opt is x or torch.equal(out_opt, x)
