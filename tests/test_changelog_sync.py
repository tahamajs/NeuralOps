#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Changelog synchronization and filesystem claim integrity (F-73, F-74).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_changelog_docs_sync() -> None:
    """Verify docs/changelog.md is bit-identical and synchronized with root CHANGELOG.md."""
    res = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "docs" / "sync_changelog.py"), "--check"],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"sync_changelog.py --check failed:\n{res.stderr}\n{res.stdout}"


def test_changelog_versions_present() -> None:
    """Verify versions 2.0.4 and 2.0.5 have explicit entries in CHANGELOG.md."""
    changelog_content = (ROOT_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [2.0.5]" in changelog_content, "Version 2.0.5 missing from CHANGELOG.md"
    assert "## [2.0.4]" in changelog_content, "Version 2.0.4 missing from CHANGELOG.md"


def test_changelog_filesystem_claims() -> None:
    """Guard test: verify CHANGELOG does not claim operations/ was removed while it still exists on disk."""
    changelog_content = (ROOT_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
    operations_dir = ROOT_DIR / "neuralops" / "operations"

    if operations_dir.exists():
        # Must not claim it was removed in favor of ops/
        assert "removed in favour of ops/" not in changelog_content
        assert "removed in favor of ops/" not in changelog_content
        # Must explicitly note deprecation / planned removal
        assert "operations/ deprecated, removal planned" in changelog_content

    status_file = ROOT_DIR / "docs" / "DIRECTORY_MODERNIZATION_STATUS.md"
    assert status_file.exists(), "docs/DIRECTORY_MODERNIZATION_STATUS.md must document migration status"
