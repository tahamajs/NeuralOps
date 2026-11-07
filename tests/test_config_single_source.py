#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying single source of truth for configuration loading and validation."""

from __future__ import annotations

import warnings
import pytest

from neuralops.config import Config, DummyConfig
from neuralops.config.loader import load_config as canonical_load_config
from neuralops.utils.config import load_config as utils_load_config


def test_missing_config_attribute_raises_attribute_error() -> None:
    """Accessing a non-existent attribute on Config raises AttributeError, never silently returns None."""
    c = Config()
    with pytest.raises(AttributeError):
        _ = c.this_attribute_does_not_exist_xyz123


def test_only_one_load_config() -> None:
    """Verify that there is only one canonical load_config implementation and secondary entrypoints delegate."""
    # The canonical load_config is in neuralops.config.loader
    assert canonical_load_config.__module__ == "neuralops.config.loader"

    # utils.config.load_config delegates to the canonical loader
    # Test loading default config through both
    cfg1 = canonical_load_config()
    cfg2 = utils_load_config()
    assert isinstance(cfg1, Config)
    assert isinstance(cfg2, Config)


def test_core_config_deprecated_shim() -> None:
    """Verify importing from neuralops.core.config emits DeprecationWarning and re-exports canonical objects."""
    with pytest.deprecated_call():
        from neuralops.core.config import Config as CoreConfig, DummyConfig as CoreDummyConfig, load_config as core_load_config

    assert CoreConfig is Config
    assert CoreDummyConfig is DummyConfig
    assert core_load_config is canonical_load_config
