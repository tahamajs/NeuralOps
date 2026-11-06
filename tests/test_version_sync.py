#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for version synchronization and source-of-truth consistency."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_all_version_sources_agree() -> None:
    """Verify that all metadata files agree on the canonical version 2.0.6."""
    canonical_version = "2.0.6"

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
    assert f"neuralops>={canonical_version}" in cuda_text, (
        f"pyproject.cuda.toml does not require neuralops>={canonical_version}"
    )

    # 3. neuralops.__version__
    import neuralops
    assert neuralops.__version__ == canonical_version, (
        f"neuralops.__version__ is {neuralops.__version__}, expected {canonical_version}"
    )

    # 4. CITATION.cff
    citation_text = (ROOT_DIR / "CITATION.cff").read_text(encoding="utf-8")
    m_cff = re.search(r"^version:\s*(\S+)", citation_text, re.M)
    assert m_cff is not None, "version not found in CITATION.cff"
    assert m_cff.group(1) == canonical_version, f"CITATION.cff version {m_cff.group(1)} != {canonical_version}"
    m_date = re.search(r"^date-released:\s*(\S+)", citation_text, re.M)
    assert m_date is not None, "date-released not found in CITATION.cff"
    assert m_date.group(1) == "2026-11-05", f"CITATION.cff date-released {m_date.group(1)} != 2026-11-05"

    # 5. SECURITY.md
    security_text = (ROOT_DIR / "SECURITY.md").read_text(encoding="utf-8")
    assert re.search(r"\|\s*\*\*2\.0\.x\*\*\s*\|\s*:white_check_mark:\s*Active\s*\|", security_text) is not None, (
        "SECURITY.md does not mark 2.0.x as Active"
    )

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
