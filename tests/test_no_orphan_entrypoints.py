#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying no orphan CLI entrypoints exist (F-66)."""

from __future__ import annotations

from pathlib import Path
import tomllib


def test_main_py_is_registered_or_absent() -> None:
    """Verify that neuralops/main.py is registered as a console script or deleted."""
    repo_root = Path(__file__).resolve().parent.parent
    main_py = repo_root / "neuralops" / "main.py"
    pyproject_path = repo_root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml missing"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    scripts = pyproject.get("project", {}).get("scripts", {})

    if main_py.exists():
        # If main.py exists on disk, it MUST be registered as an official entrypoint
        registered = any("neuralops.main:main" in target for target in scripts.values())
        assert registered, (
            f"neuralops/main.py exists ({main_py}) but is not registered in pyproject.toml scripts: {scripts}"
        )
    else:
        # If deleted, no scripts entry may point to it
        stale_ref = [name for name, target in scripts.items() if "neuralops.main" in target]
        assert not stale_ref, f"Stale pyproject.toml scripts pointing to deleted main.py: {stale_ref}"
