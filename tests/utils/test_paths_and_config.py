#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for paths, configuration, and documentation discovery in utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralops.utils.config import save_config, load_config
from neuralops.utils.hf_git_exporter import find_documentation_file, find_git_root
from neuralops.utils.paths import get_project_root, resolve_path


def test_get_project_root_contains_pyproject():
    """Verify that get_project_root locates the actual project repository root."""
    root = get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists(), f"pyproject.toml not found in {root}"
    assert (root / "neuralops").exists(), f"neuralops package not found in {root}"


def test_resolve_path_resolves_relative_to_root():
    """Verify resolve_path constructs valid absolute paths under root."""
    resolved = resolve_path("pyproject.toml")
    assert Path(resolved).exists()
    assert Path(resolved).name == "pyproject.toml"


def test_save_config_yaml(tmp_path: Path):
    """Verify that save_config serializes yaml without NameError (F821 fix)."""
    import yaml

    cfg = {"model": "gpt2", "rank": 64, "training": {"batch_size": 16}}
    yaml_file = tmp_path / "test_config.yaml"
    save_config(cfg, str(yaml_file))

    assert yaml_file.exists()
    with open(yaml_file) as f:
        loaded = yaml.safe_load(f)
    assert loaded["model"] == "gpt2"
    assert loaded["rank"] == 64
    assert loaded["training"]["batch_size"] == 16


def test_find_git_root_and_documentation():
    """Verify git root resolution and documentation discovery in Docs/."""
    git_root = find_git_root()
    assert git_root.exists()
    assert (git_root / "pyproject.toml").exists()

    # Verify documentation lookup for existing attention or compression module
    doc = find_documentation_file("full_attention")
    if doc is not None:
        assert doc.exists()
