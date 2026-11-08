#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for version synchronization and source-of-truth consistency."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_all_version_sources_agree() -> None:
    """Verify that all metadata files agree on the canonical version 2.1.0."""
    canonical_version = "2.1.0"

    # 1. pyproject.toml
    pyproject_text = (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    m_pyproj = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject_text, re.M)
    assert m_pyproj is not None, "Version not found in pyproject.toml"
    assert m_pyproj.group(1) == canonical_version, f"pyproject.toml version {m_pyproj.group(1)} != {canonical_version}"

    # 2. pyproject.cuda.toml
    cuda_text = (ROOT_DIR / "pyproject.cuda.toml").read_text(encoding="utf-8")
    m_cuda = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', cuda_text, re.M)
    assert m_cuda is not None, "Version not found in pyproject.cuda.toml"
    assert m_cuda.group(1) == canonical_version, f"pyproject.cuda.toml version {m_cuda.group(1)} != {canonical_version}"
    assert (
        f"neuralops>={canonical_version}" in cuda_text
    ), f"pyproject.cuda.toml does not require neuralops>={canonical_version}"

    # 3. neuralops.__version__
    import neuralops

    assert (
        neuralops.__version__ == canonical_version
    ), f"neuralops.__version__ is {neuralops.__version__}, expected {canonical_version}"

    # 4. CITATION.cff
    citation_text = (ROOT_DIR / "CITATION.cff").read_text(encoding="utf-8")
    m_cff = re.search(r"^version:\s*(\S+)", citation_text, re.M)
    assert m_cff is not None, "version not found in CITATION.cff"
    assert m_cff.group(1) == canonical_version, f"CITATION.cff version {m_cff.group(1)} != {canonical_version}"
    m_date = re.search(r"^date-released:\s*(\S+)", citation_text, re.M)
    assert m_date is not None, "date-released not found in CITATION.cff"
    assert m_date.group(1) == "2026-11-08", f"CITATION.cff date-released {m_date.group(1)} != 2026-11-08"

    # 5. SECURITY.md
    security_text = (ROOT_DIR / "SECURITY.md").read_text(encoding="utf-8")
    assert (
        re.search(r"\|\s*\*\*2\.1\.x\*\*\s*\|\s*:white_check_mark:\s*Active\s*\|", security_text) is not None
    ), "SECURITY.md does not mark 2.1.x as Active"

    # 6. README.md
    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert f"v{canonical_version}" in readme_text, f"README.md does not mention v{canonical_version}"
    assert "v2.0.1-gpu-verified" not in readme_text, "Stale v2.0.1-gpu-verified still present in README.md"
    assert "NeuralOps 1.9.0" not in readme_text, "Stale 'NeuralOps 1.9.0' still present in README.md"

    # 7. neuralops/README.md
    nops_readme_text = (ROOT_DIR / "neuralops" / "README.md").read_text(encoding="utf-8")
    assert "v1.6.0" not in nops_readme_text, "Stale v1.6.0 tier structure string in neuralops/README.md"
    assert f"v{canonical_version}" in nops_readme_text, f"neuralops/README.md does not mention v{canonical_version}"


def test_sync_version_check_passes() -> None:
    """Verify that sync_version.py --check exits 0 with zero drift detected."""
    res = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "utils" / "sync_version.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"sync_version.py --check failed (rc={res.returncode}):\n{res.stdout}\n{res.stderr}"


def test_security_md_supports_current_major() -> None:
    """Verify SECURITY.md support table has 2.1.x active and links to NeuralOps repo."""
    security_text = (ROOT_DIR / "SECURITY.md").read_text(encoding="utf-8")
    assert "| **2.1.x** | :white_check_mark: Active |" in security_text, "2.1.x not marked as active in SECURITY.md"
    assert "SVD_linear_Attention" not in security_text, "Stale SVD_linear_Attention link remains in SECURITY.md"
    assert (
        "github.com/tahamajs/NeuralOps/security/advisories/new" in security_text
    ), "Canonical NeuralOps security advisories URL not found in SECURITY.md"
    # Ensure no duplicate 1.7.x row
    assert security_text.count("1.7.x") <= 1, "Duplicate 1.7.x row remains in SECURITY.md"


def test_cuda_pyproject_forms_match_main() -> None:
    """Verify pyproject.cuda.toml matches pyproject.toml forms and has no build-time torch."""
    import tomllib

    with open(ROOT_DIR / "pyproject.toml", "rb") as f:
        main_data = tomllib.load(f)
    with open(ROOT_DIR / "pyproject.cuda.toml", "rb") as f:
        cuda_data = tomllib.load(f)

    # Verify build-system requires does not contain torch
    cuda_build_requires = cuda_data.get("build-system", {}).get("requires", [])
    assert not any("torch" in req for req in cuda_build_requires), (
        f"pyproject.cuda.toml build-system.requires should not contain torch: {cuda_build_requires}"
    )

    # Verify readme form
    assert cuda_data.get("project", {}).get("readme") == main_data.get("project", {}).get("readme"), (
        "pyproject.cuda.toml readme form does not match pyproject.toml"
    )

    # Verify license form
    assert cuda_data.get("project", {}).get("license") == main_data.get("project", {}).get("license"), (
        "pyproject.cuda.toml license form does not match pyproject.toml"
    )
    assert cuda_data.get("project", {}).get("license-files") == main_data.get("project", {}).get("license-files"), (
        "pyproject.cuda.toml license-files form does not match pyproject.toml"
    )

