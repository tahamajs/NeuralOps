# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Core Registry counts, taxonomy integrity, and discovery contracts.
"""

from __future__ import annotations

from collections import defaultdict
import inspect
import pytest

from neuralops.core.discovery import discover_all_modules
from neuralops.core.exceptions import RegistryError
from neuralops.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    Registry,
    get_compressor,
    list_attentions,
    list_compressors,
)


def test_distinct_attention_class_count():
    """Verify distinct attention class count equals the audited honest count (114-115)."""
    classes = ATTENTION_REGISTRY.distinct_classes()
    assert len(classes) in (114, 115), f"Expected 114 or 115 distinct attention classes, got {len(classes)}"

    dict_classes = {v for v in ATTENTION_REGISTRY._dict.values() if isinstance(v, type)}
    assert len(dict_classes) in (114, 115)


def test_compressor_canonical_count():
    """Verify canonical compressor count equals or exceeds the audited baseline count (82)."""
    COMPRESSOR_REGISTRY._load_all()
    canonical_keys = COMPRESSOR_REGISTRY.canonical_keys()
    assert len(canonical_keys) >= 82, f"Expected at least 82 canonical compressors, got {len(canonical_keys)}"


def test_flagship_compressors_resolve_to_classes():
    """Verify major compression engines resolve to genuine classes."""
    for name in ["gptq", "awq", "smoothquant", "fp8", "quarot", "asvd", "wanda", "sparsegpt"]:
        comp_cls = get_compressor(name)
        assert comp_cls is not None, f"Compressor '{name}' failed to resolve"
        assert inspect.isclass(comp_cls), f"Compressor '{name}' must resolve to class, got {type(comp_cls)}"


def test_no_class_has_two_canonical_keys():
    """Verify every registered attention class has exactly one canonical key."""
    ATTENTION_REGISTRY._load_all()
    class_to_keys = defaultdict(list)
    for k in ATTENTION_REGISTRY.canonical_keys():
        v = ATTENTION_REGISTRY._dict[k]
        if isinstance(v, type):
            class_to_keys[v].append(k)

    multi_keys = {cls: keys for cls, keys in class_to_keys.items() if len(keys) > 1}
    assert not multi_keys, f"Found classes with >1 canonical keys: {multi_keys}"


def test_register_duplicate_canonical_key_raises():
    """Verify registering a duplicate canonical key on a different class raises RegistryError."""
    reg = Registry("test_duplicate_reg")

    class ModelA:
        pass

    class ModelB:
        pass

    reg.register("key_a")(ModelA)
    # Re-registering identical object on same key is idempotent
    reg.register("key_a")(ModelA)

    # Registering differing class on existing key must raise RegistryError
    with pytest.raises(RegistryError):
        reg.register("key_a")(ModelB)


def test_alias_cannot_collide_with_canonical_key():
    """Verify registering an alias that collides with an existing canonical key raises RegistryError."""
    reg = Registry("test_alias_collision_reg")

    class ModelA:
        pass

    reg.register("canonical_key")(ModelA)

    class ModelB:
        pass

    with pytest.raises(RegistryError):
        reg.register("canonical_key", is_alias=True)(ModelB)


def test_public_list_helpers():
    """Verify list_attentions() and list_compressors() return valid non-empty lists."""
    attentions = list_attentions()
    assert isinstance(attentions, list)
    assert len(attentions) >= 114

    compressors = list_compressors()
    assert isinstance(compressors, list)
    assert len(compressors) >= 82


def test_core_discovery_contracts():
    """Verify discover_all_modules() contracts for metadata attributes."""
    modules = discover_all_modules()
    assert len(modules) >= 197
    for mod in modules:
        assert mod.name
        assert mod.type in ("attention", "compressor")
        assert isinstance(mod.supports_causal, bool)
        assert isinstance(mod.supports_gqa, bool)
        assert isinstance(mod.supports_rope, bool)
        assert isinstance(mod.requires_cuda, bool)
