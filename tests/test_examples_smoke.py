#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying compilation, truthfulness, and dynamic counts of examples suite (F-77)."""

from __future__ import annotations

from pathlib import Path
import py_compile
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def get_all_numbered_examples() -> list[Path]:
    """Retrieve all 35 numbered example scripts."""
    scripts = sorted(EXAMPLES_DIR.glob("[0-9][0-9]_*.py"))
    return scripts


def test_all_35_numbered_examples_present() -> None:
    """Verify that exactly 35 numbered demonstration scripts exist in examples/."""
    scripts = get_all_numbered_examples()
    assert len(scripts) == 35, f"Expected 35 numbered example scripts, found {len(scripts)}: {[s.name for s in scripts]}"


@pytest.mark.parametrize("script_path", get_all_numbered_examples(), ids=lambda p: p.name)
def test_example_script_compiles(script_path: Path) -> None:
    """Verify that each numbered example compiles with python -m py_compile (Q6 compile-all tier)."""
    try:
        py_compile.compile(str(script_path), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"Compilation failure in {script_path.name}: {exc}")


def test_example_33_reads_triton_count_dynamically() -> None:
    """Verify example 33 queries the runtime catalog instead of asserting a hardcoded operator count."""
    script_path = EXAMPLES_DIR / "33_triton_26_hardware_kernels.py"
    assert script_path.exists(), "33_triton_26_hardware_kernels.py missing"
    content = script_path.read_text(encoding="utf-8")

    # Assert no stale hardcoded claims
    assert "62+ registered operators" not in content, "Stale '62+ registered operators' found in example 33"
    assert "len(catalog) >= 26" not in content, "Hardcoded '>=' assertion found in example 33"

    # Assert dynamic catalog usage
    assert "TRITON_KERNEL_CATALOG" in content, "Example 33 must inspect TRITON_KERNEL_CATALOG"
    assert "read dynamically from registry" in content or "count is read dynamically" in content


def test_examples_readme_states_mock_disclaimer_and_current_version() -> None:
    """Verify examples/README.md mentions mock models and does not contain stale version tags."""
    readme_path = EXAMPLES_DIR / "README.md"
    assert readme_path.exists(), "examples/README.md missing"
    text = readme_path.read_text(encoding="utf-8")

    assert "v2.0.3" not in text, "Stale v2.0.3 found in examples/README.md"
    assert "mock_models" in text, "Mock models disclaimer missing in examples/README.md"
    assert "offline_data" in text, "Offline data disclaimer missing in examples/README.md"
