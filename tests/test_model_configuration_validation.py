#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Configuration validation tests for NeuralOps model architectures."""

import pytest

from neuralops.models.configuration import NeuralOpsConfig


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_size": 0}, "hidden_size must be positive"),
        ({"num_attention_heads": 0}, "num_attention_heads must be positive"),
        ({"num_key_value_heads": 0}, "num_key_value_heads must be positive"),
        (
            {"hidden_size": 10, "num_attention_heads": 3},
            "hidden_size must be divisible",
        ),
        ({"head_dim": 0}, "head_dim must be positive"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
        ({"num_hidden_layers": 0}, "num_hidden_layers must be positive"),
        ({"intermediate_size": 0}, "intermediate_size must be positive"),
        ({"max_position_embeddings": 0}, "max_position_embeddings must be positive"),
    ],
)
def test_invalid_attention_dimensions_fail_early(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NeuralOpsConfig(**kwargs)


def test_explicit_head_dimension_can_differ_from_hidden_width():
    config = NeuralOpsConfig(hidden_size=10, num_attention_heads=3, head_dim=4)
    assert config.head_dim == 4
