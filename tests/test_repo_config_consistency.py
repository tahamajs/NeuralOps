#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for consolidated repository configuration in pyproject.toml."""

from __future__ import annotations

from pathlib import Path
import tomllib

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_standalone_ruff_or_pytest_config() -> None:
    """Verify that standalone ruff.toml and pytest.ini have been deleted."""
    ruff_toml = ROOT_DIR / "ruff.toml"
    pytest_ini = ROOT_DIR / "pytest.ini"
    assert not ruff_toml.exists(), f"ruff.toml still exists at {ruff_toml}"
    assert not pytest_ini.exists(), f"pytest.ini still exists at {pytest_ini}"


def test_pytest_markers_complete() -> None:
    """Verify all required markers and options are consolidated into pyproject.toml."""
    pyproject_path = ROOT_DIR / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert pytest_opts, "No [tool.pytest.ini_options] in pyproject.toml"

    markers = pytest_opts.get("markers", [])
    marker_str = " ".join(markers)

    expected_markers = [
        "gpu",
        "triton",
        "integration",
        "regression",
        "performance",
        "vllm",
        "sglang",
        "trt_llm",
        "slow",
    ]
    for em in expected_markers:
        assert any(m.startswith(em) for m in markers), f"Marker '{em}' missing from pyproject.toml markers: {markers}"

    # Verify slow threshold is 10s (winning value from pytest.ini)
    assert any("10s" in m or "10 seconds" in m for m in markers if m.startswith("slow")), (
        f"slow marker threshold is not 10s in pyproject.toml: {marker_str}"
    )

    # Verify addopts
    addopts = pytest_opts.get("addopts", "")
    assert "-ra" in addopts and "--strict-markers" in addopts, f"addopts missing expected flags: {addopts}"

    # Verify filterwarnings
    filterwarnings = pytest_opts.get("filterwarnings", [])
    assert "error" in filterwarnings, f"'error' not found in filterwarnings: {filterwarnings}"
    assert len(filterwarnings) >= 15, f"Expected full filterwarnings list, got {len(filterwarnings)}"

    # Verify testpaths and python_files
    assert pytest_opts.get("testpaths") == ["tests"]
    assert pytest_opts.get("python_files") == ["test_*.py"]


def test_ci_has_no_duplicate_test_workflow() -> None:
    """Verify that duplicate .github/workflows/test.yml has been deleted."""
    duplicate_test_workflow = ROOT_DIR / ".github" / "workflows" / "test.yml"
    assert not duplicate_test_workflow.exists(), (
        f"Duplicate test workflow still exists at {duplicate_test_workflow}"
    )


def test_ci_runs_full_suite_or_documented_subset() -> None:
    """Verify that CI workflow runs test suite, mypy, and ruff linting."""
    import yaml

    ci_yaml_path = ROOT_DIR / ".github" / "workflows" / "ci.yml"
    assert ci_yaml_path.exists(), f"ci.yml not found at {ci_yaml_path}"

    with open(ci_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jobs = data.get("jobs", {})
    assert "lint" in jobs, "CI missing 'lint' job"
    assert "mypy" in jobs, "CI missing 'mypy' job"

    # Verify at least one test job runs pytest tests/ or documented subset
    test_commands: list[str] = []
    for job_name, job_def in jobs.items():
        steps = job_def.get("steps", [])
        for step in steps:
            run_cmd = step.get("run", "")
            if "pytest" in run_cmd:
                test_commands.append(run_cmd)

    assert any("pytest tests/" in cmd for cmd in test_commands), (
        f"No CI job found running full suite or documented subset with 'pytest tests/': {test_commands}"
    )
