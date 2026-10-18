"""Part 10 Verification: Registry Health & Duplicate Consolidation."""
from pathlib import Path

import pytest
from neuralops.core.registry import (ATTENTION_REGISTRY, COMPRESSOR_REGISTRY,
                                     is_helper_module)


def test_registry_size_healthy():
    assert len(ATTENTION_REGISTRY.keys()) >= 15
    assert len(COMPRESSOR_REGISTRY.keys()) >= 5

def test_helper_module_exclusions():
    assert is_helper_module("adaptive_svd_factory")
    assert is_helper_module("adaptive_svd_inject")
    assert not is_helper_module("adaptive_svd")
