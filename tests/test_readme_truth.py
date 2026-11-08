#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying README honesty, single version source of truth, and no file:// links (F-78)."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib
import pytest

from scripts.docs.check_readme_sections import check_readme_sections

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_canonical_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml missing"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def test_both_readmes_state_same_canonical_version() -> None:
    """Verify root README.md and neuralops/README.md both state the current canonical package version."""
    version = get_canonical_version()

    root_readme = REPO_ROOT / "README.md"
    pkg_readme = REPO_ROOT / "neuralops" / "README.md"

    assert root_readme.exists(), "Root README.md missing"
    assert pkg_readme.exists(), "neuralops/README.md missing"

    root_text = root_readme.read_text(encoding="utf-8")
    pkg_text = pkg_readme.read_text(encoding="utf-8")

    assert f"v{version}" in root_text, f"Root README.md does not mention canonical v{version}"
    assert f"v{version}" in pkg_text, f"neuralops/README.md does not mention canonical v{version}"
    assert "v1.6.0" not in pkg_text, "Stale v1.6.0 found in neuralops/README.md"


def test_no_file_uri_in_documentation() -> None:
    """Verify that no file:// URI links exist in README.md, neuralops/README.md, or docs/ markdown files."""
    targets: list[Path] = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "neuralops" / "README.md",
    ]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.exists():
        targets.extend(docs_dir.glob("**/*.md"))

    violations: list[str] = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        matches = re.findall(r"file://[^\s\)\"'>]+", text)
        if matches:
            violations.append(f"{path.relative_to(REPO_ROOT)}: {matches}")

    assert not violations, "Prohibited file:// URIs detected in documentation:\n" + "\n".join(violations)


def test_readme_section_numbering_continuous_and_deduplicated() -> None:
    """Verify that README.md has no duplicate section numbers and forms a continuous monotonic sequence."""
    ret = check_readme_sections(REPO_ROOT / "README.md")
    assert ret == 0, "check_readme_sections reported duplicate or malformed numbered sections in README.md"
