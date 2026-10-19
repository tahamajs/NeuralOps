# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for deterministic attention mode."""
from neuralops.core.attention_utils import enable_deterministic_attention, is_deterministic_attention


def test_deterministic_attention_toggle():
    assert not is_deterministic_attention()
    prev = enable_deterministic_attention(True)
    assert not prev
    assert is_deterministic_attention()
    prev2 = enable_deterministic_attention(False)
    assert prev2
    assert not is_deterministic_attention()
