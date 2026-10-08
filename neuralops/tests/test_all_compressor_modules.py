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
