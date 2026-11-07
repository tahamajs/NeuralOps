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


def test_subpackages_match_filesystem():
    """Verify that every entry in neuralops._SUBPACKAGES exists as a directory with __init__.py."""
    from pathlib import Path

    pkg_root = Path(neuralops.__file__).resolve().parent
    for subpkg in neuralops._SUBPACKAGES:
        subpkg_dir = pkg_root / subpkg
        assert subpkg_dir.is_dir(), f"_SUBPACKAGES entry '{subpkg}' is not a directory at {subpkg_dir}"
        assert (subpkg_dir / "__init__.py").exists(), (
            f"_SUBPACKAGES entry '{subpkg}' lacks __init__.py at {subpkg_dir / '__init__.py'}"
        )



def test_getattr_missing_raises_attribute_error_not_module_not_found():
    """Verify missing/phantom subpackages raise AttributeError and hasattr returns False cleanly."""
    assert getattr(neuralops, "experiment_logging", None) is None
    assert getattr(neuralops, "validator", None) is None
    assert getattr(neuralops, "non_existent_symbol_xyz", None) is None

    assert hasattr(neuralops, "validator") is False
    assert hasattr(neuralops, "experiment_logging") is False
    assert hasattr(neuralops, "tests") is False

    with pytest.raises(AttributeError):
        _ = neuralops.non_existent_subpackage_xyz


def test_deprecated_alias_warns_and_resolves():
    """Accessing neuralops.attentions warns with DeprecationWarning and resolves to neuralops.layers."""
    import warnings
    import neuralops.layers as layers

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        # Ensure clean lookup by removing cache if present
        if "attentions" in neuralops.__dict__:
            del neuralops.__dict__["attentions"]
        mod = neuralops.attentions
        assert mod is layers
        deprecation_warnings = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message)


def test_phantom_subpackage_raises_attribute_error():
    """Verify phantom subpackages raise AttributeError instead of ModuleNotFoundError."""
    with pytest.raises(AttributeError):
        _ = neuralops.validator
    with pytest.raises(AttributeError):
        _ = neuralops.experiment_logging

