#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying that relative markdown links in documentation resolve to valid files."""

from pathlib import Path
import re
import pytest

DOCS_ROOT = Path("docs")


def _get_markdown_files():
    return sorted(list(DOCS_ROOT.glob("*.md")) + list(DOCS_ROOT.glob("api/*.md")))


@pytest.mark.parametrize("md_path", _get_markdown_files())
def test_relative_markdown_links_resolve(md_path: Path):
    """Verify that all relative markdown links in docs/*.md resolve to existing files."""
    content = md_path.read_text(encoding="utf-8")
    links = re.findall(r"\[.*?\]\((?!http|#|mailto:)(.*?)\)", content)

    for link in links:
        # Strip anchor if present
        clean_link = link.split("#")[0].strip()
        if not clean_link:
            continue

        target = (md_path.parent / clean_link).resolve()
        assert target.exists(), f"Broken relative link '{link}' in {md_path} (resolved to {target})"
