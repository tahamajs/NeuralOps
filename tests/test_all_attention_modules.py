#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke tests for ALL registered attention modules."""

import pytest
import warnings
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", sorted(ATTENTION_REGISTRY.keys()))
def test_attention_module_importable(module_name):
    """Verify that every registered attention key can be instantiated or retrieved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = ATTENTION_REGISTRY.get(module_name)
    assert cls is not None
