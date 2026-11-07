#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying runner sweep execution and clean module structure (F-67)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

from neuralops.runners.run_sweep import REGISTRY_PATH, main


def test_run_sweep_does_not_rewrite_sys_argv() -> None:
    """Verify that calling run_sweep.main() does not mutate global sys.argv (F-67)."""
    original_argv = list(sys.argv)

    with patch("neuralops.runners.run_sweep.run_orchestrator") as mock_orch:
        # Pass simulated CLI arguments directly to main()
        main(["--module", "cur_linear", "--epochs", "5"])

        assert mock_orch.called
        call_args = mock_orch.call_args
        assert call_args[0][0] == "cur_linear"
        # Check that extra CLI args were passed explicitly
        assert "--epochs" in call_args[1].get("extra_cli_args", [])

    # Assert sys.argv was not modified
    assert sys.argv == original_argv, f"sys.argv was mutated: original={original_argv}, current={sys.argv}"


def test_module_sweeps_absent_or_implemented() -> None:
    """Verify runners/module_sweeps is either absent or contains real implementations beyond __init__.py."""
    repo_root = Path(__file__).resolve().parent.parent
    module_sweeps_dir = repo_root / "neuralops" / "runners" / "module_sweeps"

    if module_sweeps_dir.exists():
        py_files = [p.name for p in module_sweeps_dir.glob("*.py") if p.name != "__init__.py"]
        assert len(py_files) > 0, (
            f"Empty subpackage: {module_sweeps_dir} contains only __init__.py with no real module sweeps"
        )


def test_sweep_registry_path_is_package_relative_and_exists() -> None:
    """Verify sweep_registry.json is package-relative and exists on disk."""
    assert REGISTRY_PATH.is_absolute()
    assert REGISTRY_PATH.exists(), f"sweep_registry.json missing at {REGISTRY_PATH}"
    assert REGISTRY_PATH.stat().st_size > 1000, f"sweep_registry.json too small: {REGISTRY_PATH.stat().st_size}"
