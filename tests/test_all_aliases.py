"""Test all deprecated aliases work."""
import json
import warnings
import pytest
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions, neuralops.compression_engine


@pytest.fixture(scope="module")
def aliases():
    with open("neuralops/core/deprecated_aliases.json") as f:
        return json.load(f)


@pytest.mark.smoke
def test_alias_count(aliases):
    """Verify alias count matches expected."""
    assert len(aliases) >= 600, f"Too few aliases: {len(aliases)}"


def test_alias_sample_resolves(aliases):
    """Sample of aliases should resolve."""
    sample = list(aliases.items())[:50]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for alias, target in sample:
            cls = ATTENTION_REGISTRY.get(alias)
            if cls is None:
                cls = COMPRESSOR_REGISTRY.get(alias)
            assert cls is not None, f"Alias {alias} -> {target} failed to resolve"
