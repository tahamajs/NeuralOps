#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for wheel packaging integrity and console scripts entry point resolution."""

from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import sys
import zipfile


def test_console_scripts_resolve_after_wheel_install() -> None:
    """Assert that all four declared console scripts resolve to importable callables."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        console_eps = {ep.name: ep for ep in eps.select(group="console_scripts")}
    else:
        console_eps = {ep.name: ep for ep in eps if getattr(ep, "group", None) == "console_scripts"}

    expected_scripts = ("neuralops", "neuralops-cli", "nops", "nops-cli")
    for script_name in expected_scripts:
        assert script_name in console_eps, f"Console script '{script_name}' missing from package metadata"
        target_fn = console_eps[script_name].load()
        assert callable(target_fn), f"Target '{target_fn}' for '{script_name}' is not callable"


def test_benchmarks_not_excluded_from_wheel(tmp_path: pathlib.Path) -> None:
    """Build the wheel into tmp_path and verify neuralops/benchmarks/cli.py is packaged."""
    res = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Wheel build failed with stderr: {res.stderr}"

    wheel_files = list(tmp_path.glob("*.whl"))
    assert len(wheel_files) == 1, f"Expected 1 wheel file in {tmp_path}, found: {wheel_files}"

    with zipfile.ZipFile(wheel_files[0], "r") as zf:
        members = set(zf.namelist())

    assert "neuralops/benchmarks/cli.py" in members, (
        f"'neuralops/benchmarks/cli.py' was excluded from built wheel {wheel_files[0].name}"
    )
    assert "neuralops/benchmarks/__init__.py" in members, (
        f"'neuralops/benchmarks/__init__.py' was excluded from built wheel {wheel_files[0].name}"
    )


def test_py_typed_shipped_in_wheel(tmp_path: pathlib.Path) -> None:
    """Build the wheel into tmp_path and verify neuralops/py.typed is packaged."""
    res = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Wheel build failed with stderr: {res.stderr}"

    wheel_files = list(tmp_path.glob("*.whl"))
    assert len(wheel_files) == 1, f"Expected 1 wheel file in {tmp_path}, found: {wheel_files}"

    with zipfile.ZipFile(wheel_files[0], "r") as zf:
        members = set(zf.namelist())

    assert "neuralops/py.typed" in members, (
        f"'neuralops/py.typed' was missing from built wheel {wheel_files[0].name}"
    )
