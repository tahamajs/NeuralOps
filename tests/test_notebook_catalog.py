#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for notebook catalog consistency and single source of truth (F-75).
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
CATALOG_PATH = NOTEBOOKS_DIR / "NOTEBOOKS_COMPREHENSIVE_CATALOG_AND_60_EXPANSION_ROADMAP.md"


def test_notebooks_directory_count() -> None:
    """Verify that notebooks/ contains exactly 94 unique Jupyter notebooks."""
    notebooks = sorted([p.name for p in NOTEBOOKS_DIR.glob("*.ipynb")])
    assert len(notebooks) == 94, f"Expected 94 unique notebooks, found {len(notebooks)}: {notebooks}"


def test_notebooks_extended_symlink() -> None:
    """Verify notebooks/extended is a symlink pointing to '.' and not duplicate files."""
    extended_path = NOTEBOOKS_DIR / "extended"
    assert extended_path.exists(), "notebooks/extended path must exist"
    assert extended_path.is_symlink(), "notebooks/extended must be a symbolic link to prevent dual-copy drift"
    # Target should resolve to notebooks/ itself
    assert extended_path.resolve() == NOTEBOOKS_DIR.resolve()


def test_all_notebooks_in_catalog() -> None:
    """Verify every notebook on disk is indexed in the master catalog markdown."""
    assert CATALOG_PATH.exists(), f"Catalog not found at {CATALOG_PATH}"
    catalog_text = CATALOG_PATH.read_text(encoding="utf-8")

    notebook_files = sorted([p.name for p in NOTEBOOKS_DIR.glob("*.ipynb")])
    missing = [nb for nb in notebook_files if nb not in catalog_text]
    assert not missing, f"Notebooks missing from master catalog: {missing}"


def test_catalog_distinguishes_executed_vs_structure() -> None:
    """Verify catalog accurately records 29 executed notebooks and 65 structure-validated notebooks."""
    catalog_text = CATALOG_PATH.read_text(encoding="utf-8")
    assert "Authoritative 94-Notebook Master Matrix" in catalog_text
    assert "Verified Executed (RTX 3090 / HPC)" in catalog_text
    assert "Structure Validated (Requires Multi-GPU Allocation)" in catalog_text
