#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for consolidated repository configuration in pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

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
    assert any(
        "10s" in m or "10 seconds" in m for m in markers if m.startswith("slow")
    ), f"slow marker threshold is not 10s in pyproject.toml: {marker_str}"

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
    assert not duplicate_test_workflow.exists(), f"Duplicate test workflow still exists at {duplicate_test_workflow}"


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

    assert any(
        "pytest tests/" in cmd for cmd in test_commands
    ), f"No CI job found running full suite or documented subset with 'pytest tests/': {test_commands}"


def test_pre_commit_config_exists_and_pins_tools() -> None:
    """Verify .pre-commit-config.yaml exists and pins ruff, black, isort, mypy."""
    import yaml

    pre_commit_path = ROOT_DIR / ".pre-commit-config.yaml"
    assert pre_commit_path.exists(), f".pre-commit-config.yaml not found at {pre_commit_path}"

    with open(pre_commit_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    repos = data.get("repos", [])
    assert repos, ".pre-commit-config.yaml has no repos configured"

    hook_ids: set[str] = set()
    repo_urls: dict[str, str] = {}
    for r in repos:
        rev = r.get("rev", "")
        repo_url = r.get("repo", "")
        for h in r.get("hooks", []):
            hid = h.get("id", "")
            hook_ids.add(hid)
            repo_urls[hid] = rev

    expected_hooks = {
        "ruff",
        "black",
        "isort",
        "mypy",
        "trailing-whitespace",
        "end-of-file-fixer",
        "check-yaml",
        "check-added-large-files",
    }
    missing = expected_hooks - hook_ids
    assert not missing, f"Missing expected hooks in .pre-commit-config.yaml: {missing}"

    # Verify that every tool hook has a non-empty pinned rev
    for h in ["ruff", "black", "isort", "mypy"]:
        assert repo_urls.get(h), f"Hook '{h}' does not have a pinned rev"


def test_mypy_target_within_supported_range() -> None:
    """Verify mypy target python_version is between requires-python and CI maximum."""
    pyproject_path = ROOT_DIR / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    mypy_version = data.get("tool", {}).get("mypy", {}).get("python_version", "")
    assert mypy_version, "No python_version in [tool.mypy]"

    # Minimum supported version is 3.10, CI maximum is 3.12
    valid_versions = {"3.10", "3.11", "3.12"}
    assert (
        mypy_version in valid_versions
    ), f"mypy python_version '{mypy_version}' is outside supported range {valid_versions}"


def test_gitignore_does_not_ignore_tracked_docs() -> None:
    """Verify that .gitignore does not ignore docs/ or audit/ and dead whitelists are absent."""
    import subprocess
    import pytest

    if not (ROOT_DIR / ".git").exists():
        pytest.skip("Test requires a git checkout")

    gitignore_path = ROOT_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore not found"

    lines = [line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()]

    # docs/ and audit/ must NOT be ignored
    assert "docs/" not in lines, "docs/ should not be ignored in .gitignore"
    assert "docs/**" not in lines, "docs/** should not be ignored in .gitignore"
    assert "audit/" not in lines, "audit/ should not be ignored in .gitignore"
    assert "audit/**" not in lines, "audit/** should not be ignored in .gitignore"

    # Dead whitelists must be gone
    assert "!ROADMAP.md" not in lines, "Dead whitelist !ROADMAP.md found in .gitignore"
    assert "!setup.py" not in lines, "Dead whitelist !setup.py found in .gitignore"

    # Verify git check-ignore returns non-zero for docs/API_REFERENCE.md
    res = subprocess.run(
        ["git", "check-ignore", "docs/API_REFERENCE.md"],
        cwd=ROOT_DIR,
        capture_output=True,
    )
    assert res.returncode != 0, "docs/API_REFERENCE.md is incorrectly ignored by git"



def test_ci_prevents_bytecode_accumulation() -> None:
    """Verify that CI workflow sets PYTHONDONTWRITEBYTECODE=1 to prevent bytecode pollution (F-16)."""
    ci_yaml_path = ROOT_DIR / ".github" / "workflows" / "ci.yml"
    assert ci_yaml_path.exists(), f"ci.yml not found at {ci_yaml_path}"

    content = ci_yaml_path.read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE" in content, (
        "PYTHONDONTWRITEBYTECODE guard is missing from .github/workflows/ci.yml"
    )


def test_no_hardcoded_user_paths_in_readme() -> None:
    """Verify README.md contains no personal /home/tahamajlesi or /Users/tahamajs paths (F-17)."""
    readme_path = ROOT_DIR / "README.md"
    assert readme_path.exists(), f"README.md not found at {readme_path}"

    lines = readme_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        # Exclude official github.com contributor URLs
        if "github.com" in line:
            continue
        assert "/home/tahamajlesi" not in line, (
            f"Hardcoded HPC path found in README.md line {idx}: {line}"
        )
        assert "/Users/tahamajs" not in line, (
            f"Hardcoded macOS user path found in README.md line {idx}: {line}"
        )


def test_no_bare_except_pass_in_layers_compression() -> None:
    """Verify zero unlogged bare 'except Exception: pass' blocks in targeted fallback files (F-26)."""
    target_rel_files = [
        "neuralops/compression/low_rank/a3_compressor.py",
        "neuralops/layers/hybrid/tri_branch_hybrid_attention.py",
        "neuralops/layers/modern/mla_attention.py",
    ]
    for rel_path in target_rel_files:
        full_path = ROOT_DIR / rel_path
        assert full_path.exists(), f"File not found: {full_path}"
        content = full_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if "except Exception:" in line:
                # Next 4 lines must not contain bare pass without logger
                following_block = "\n".join(lines[idx + 1 : min(len(lines), idx + 5)])
                if "pass" in following_block:
                    assert "logger" in following_block or "LOGGER" in following_block, (
                        f"Found unlogged bare pass in {rel_path} line {idx + 1}:\n{following_block}"
                    )
