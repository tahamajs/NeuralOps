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
