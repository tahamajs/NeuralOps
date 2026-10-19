# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke tests for ALL registered compressor modules."""

import pytest
import warnings
from neuralops.core.registry import COMPRESSOR_REGISTRY
import neuralops.compression_engine


@pytest.mark.smoke
@pytest.mark.parametrize("module_name", sorted(COMPRESSOR_REGISTRY.keys()))
def test_compressor_module_importable(module_name):
    """Verify that every registered compressor key can be retrieved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cls = COMPRESSOR_REGISTRY.get(module_name)
    assert cls is not None
