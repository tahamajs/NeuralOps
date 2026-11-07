#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import neuralops
import neuralops.api as api


def test_api_all_subset_of_init_all():
    """Verify that every export in neuralops.api.__all__ is present in neuralops.__all__."""
    api_all = set(api.__all__)
    init_all = set(neuralops.__all__)
    missing = api_all - init_all
    assert not missing, f"Names in api.__all__ missing from neuralops.__all__: {missing}"


def test_every_all_name_importable_from_neuralops():
    """Verify that every symbol in neuralops.__all__ can be accessed via getattr(neuralops, name)."""
    unreachable = []
    for name in neuralops.__all__:
        try:
            val = getattr(neuralops, name)
            assert val is not None, f"Attribute {name} resolved to None"
        except Exception as e:
            unreachable.append((name, str(e)))
    assert not unreachable, f"Unreachable symbols in neuralops.__all__: {unreachable}"


def test_four_missing_exports_now_reachable():
    """Verify the 4 previously unreachable api.py exports can be imported from top-level neuralops."""
    from neuralops import TextGenerationPipeline, heal_distributed, load_and_prepare_dataset, train_grpo

    assert callable(TextGenerationPipeline) or isinstance(TextGenerationPipeline, type)
    assert callable(heal_distributed)
    assert callable(load_and_prepare_dataset)
    assert callable(train_grpo)
