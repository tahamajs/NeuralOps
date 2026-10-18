"""Test that required documentation files exist."""
from pathlib import Path

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
]


@pytest.mark.parametrize("doc_path", REQUIRED_DOCS)
def test_doc_exists(doc_path):
    """Every required doc should exist."""
    p = Path(doc_path)
    assert p.exists(), f"Missing: {doc_path}"
    assert p.stat().st_size > 0, f"Empty: {doc_path}"
