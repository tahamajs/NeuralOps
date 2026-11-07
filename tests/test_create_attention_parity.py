#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import neuralops
import neuralops.api
import neuralops.core.registry as registry


def test_create_attention_same_object_via_all_paths():
    """Verify create_attention identity and factory equivalence across all 3 paths."""
    # 1. Function object identity
    assert neuralops.create_attention is neuralops.api.create_attention
    assert neuralops.create_attention is registry.create_attention

    # 2. Same call produces equivalent module type
    attn1 = neuralops.create_attention("linear_attention", dim=64, num_heads=4)
    attn2 = neuralops.api.create_attention("linear_attention", dim=64, num_heads=4)
    attn3 = registry.create_attention("linear_attention", dim=64, num_heads=4)

    assert type(attn1) is type(attn2)
    assert type(attn2) is type(attn3)


def test_create_compressor_identity_via_all_paths():
    """Verify create_compressor function identity across all 3 paths."""
    assert neuralops.create_compressor is neuralops.api.create_compressor
    assert neuralops.create_compressor is registry.create_compressor


def test_list_attentions_returns_sorted_list():
    """Verify list_attentions returns a sorted list."""
    res = registry.list_attentions()
    assert isinstance(res, list)
    assert len(res) > 0
    assert res == sorted(res)

    api_res = neuralops.api.list_attentions()
    assert isinstance(api_res, list)
    assert api_res == sorted(api_res)


def test_create_attention_invalid_num_heads_raises_value_error():
    """Verify that if num_heads does not divide dim evenly, ValueError is raised."""
    with pytest.raises(ValueError, match="must divide dim"):
        neuralops.create_attention("linear_attention", dim=65, num_heads=4)


def test_create_attention_composable_still_works():
    """Verify that composable attention creation still works cleanly."""
    comp = neuralops.create_attention("composable", dim=64, num_heads=4)
    assert comp is not None
