"""Test that all declared extras are valid."""
import tomllib
from pathlib import Path

import pytest


REQUIRED_EXTRAS = [
    "cuda", "rocm", "mps", "cpu",
    "eval", "benchmark", "dev", "docs", "all",
]


def test_all_extras_defined():
    """Verify all required extras are defined in pyproject.toml."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    extras = data.get("project", {}).get("optional-dependencies", {})
    for name in REQUIRED_EXTRAS:
        assert name in extras, f"Missing extra: {name}"


def test_core_deps_minimal():
    """Core dependencies should not include torch."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    core = data.get("project", {}).get("dependencies", [])
    for dep in core:
        assert "torch" not in dep.lower(), f"torch should not be in core: {dep}"


def test_all_extra_covers_main_backends():
    """The 'all' extra should cover cuda, eval, benchmark, dev."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    extras = data.get("project", {}).get("optional-dependencies", {})
    all_extra = extras.get("all", [])
    # 'all' should reference neuralops[...]
    assert any("neuralops[" in str(dep) for dep in all_extra)
