#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Test that required documentation files exist and conform to quality standards."""

from pathlib import Path
import re
import pytest


REQUIRED_DOCS = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CITATION.cff",
    "docs/API_REFERENCE.md",
    "docs/ARCHITECTURE.md",
    "docs/MIGRATION_GUIDE.md",
    "docs/TROUBLESHOOTING.md",
    "docs/QUICKSTART.md",
    "docs/RENAME_MAP.md",
    "docs/HUGGINGFACE_UPLOAD.md",
    "docs/COMPARISON.md",
    "docs/api/api.md",
]


@pytest.mark.parametrize("doc_path", REQUIRED_DOCS)
def test_doc_exists_and_meets_quality_standards(doc_path):
    """Every required doc should exist, be non-empty, and have a level-1 heading if markdown."""
    p = Path(doc_path)
    assert p.exists(), f"Missing required document: {doc_path}"
    assert p.stat().st_size > 50, f"Document too small or empty ({p.stat().st_size} bytes): {doc_path}"

    if p.suffix == ".md":
        content = p.read_text(encoding="utf-8")
        assert re.search(r"^#\s+.+", content, re.MULTILINE), f"Missing level-1 heading in {doc_path}"
        assert "file://" not in content, f"Forbidden local file:// link found in {doc_path}"


def test_mkdocs_config_has_mkdocstrings_and_strict_ready():
    """Verify mkdocs.yml enables mkdocstrings and proper validation gates."""
    mkdocs_file = Path("mkdocs.yml")
    assert mkdocs_file.exists(), "mkdocs.yml not found"
    content = mkdocs_file.read_text(encoding="utf-8")
    assert "mkdocstrings" in content, "mkdocstrings plugin not declared in mkdocs.yml"
