#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for scripts/ci/check_test_layout.py."""

from pathlib import Path
import pytest

from scripts.ci.check_test_layout import check_test_layout, main


def test_check_test_layout_passes_on_real_repo():
    """Verify that check_test_layout passes on the actual NeuralOps repository."""
    repo_root = Path(__file__).resolve().parent.parent
    ok, errors = check_test_layout(repo_root)
    assert ok is True
    assert len(errors) == 0


def test_check_test_layout_fails_on_missing_test_directory(tmp_path: Path):
    """Verify check_test_layout detects a missing tests/<subpkg>/ directory."""
    pkg_dir = tmp_path / "neuralops"
    tests_dir = tmp_path / "tests"
    pkg_dir.mkdir()
    tests_dir.mkdir()

    # Create subpackage foo
    foo_pkg = pkg_dir / "foo"
    foo_pkg.mkdir()
    (foo_pkg / "__init__.py").touch()

    # Do not create tests/foo/
    ok, errors = check_test_layout(tmp_path)
    assert ok is False
    assert any("Missing test directory: tests/foo/" in e for e in errors)


def test_check_test_layout_fails_on_empty_test_directory(tmp_path: Path):
    """Verify check_test_layout detects an empty tests/<subpkg>/ directory."""
    pkg_dir = tmp_path / "neuralops"
    tests_dir = tmp_path / "tests"
    pkg_dir.mkdir()
    tests_dir.mkdir()

    bar_pkg = pkg_dir / "bar"
    bar_pkg.mkdir()
    (bar_pkg / "__init__.py").touch()

    # Create empty tests/bar/
    bar_test = tests_dir / "bar"
    bar_test.mkdir()

    ok, errors = check_test_layout(tmp_path)
    assert ok is False
    assert any("contains no test_*.py files" in e for e in errors)

    # Now add a test file, should pass
    (bar_test / "test_bar.py").touch()
    ok_fixed, errors_fixed = check_test_layout(tmp_path)
    assert ok_fixed is True
    assert len(errors_fixed) == 0
