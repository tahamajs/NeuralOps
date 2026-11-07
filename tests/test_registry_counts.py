# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for honest registry counts and canonical key uniqueness (F-18)."""

from collections import defaultdict
import pytest

from neuralops.core.exceptions import RegistryError
from neuralops.core.registry import ATTENTION_REGISTRY, Registry


def test_distinct_attention_class_count():
    """Verify distinct attention class count equals the audited honest count (113)."""
    classes = ATTENTION_REGISTRY.distinct_classes()
    assert len(classes) == 113

    # Also verify through _dict values directly
    dict_classes = {v for v in ATTENTION_REGISTRY._dict.values() if isinstance(v, type)}
    assert len(dict_classes) == 113


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

    # First registration succeeds
    reg.register("key_a")(ModelA)

    # Re-registering identical object on same key is idempotent
    reg.register("key_a")(ModelA)

    # Registering duplicate canonical key on different class raises RegistryError
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("key_a")(ModelB)

    # Registering an alias onto an existing canonical key raises RegistryError
    with pytest.raises(RegistryError, match="collides with canonical key"):
        reg.register("key_a", is_alias=True)(ModelB)

    # Registering a canonical key when key exists as an alias raises RegistryError
    reg.register("alias_b", is_alias=True)(ModelB)
    with pytest.raises(RegistryError, match="collides with an existing alias"):
        reg.register("alias_b")(ModelA)
