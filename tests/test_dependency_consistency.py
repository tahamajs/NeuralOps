#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for dependency consistency, lockfile integrity, and MANIFEST.in."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_no_editable_git_dependency_in_lock() -> None:
    """Verify that requirements.lock.txt contains no editable git dependencies."""
    lock_path = ROOT_DIR / "requirements.lock.txt"
    assert lock_path.exists(), f"requirements.lock.txt not found at {lock_path}"

    content = lock_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        assert not line.startswith("-e git+"), (
            f"Found editable git dependency in requirements.lock.txt: {line}"
        )
        assert "git+https://" not in line, (
            f"Found git VCS dependency in requirements.lock.txt: {line}"
        )


def test_core_deps_present_in_requirements() -> None:
    """Verify einops and safetensors are present in requirements.txt and lock."""
    req_path = ROOT_DIR / "requirements.txt"
    lock_path = ROOT_DIR / "requirements.lock.txt"
    assert req_path.exists(), f"requirements.txt not found at {req_path}"
    assert lock_path.exists(), f"requirements.lock.txt not found at {lock_path}"

    req_content = req_path.read_text(encoding="utf-8")
    assert "einops" in req_content, "einops missing from requirements.txt"
    assert "safetensors" in req_content, "safetensors missing from requirements.txt"

    lock_content = lock_path.read_text(encoding="utf-8")
    assert "einops" in lock_content, "einops missing from requirements.lock.txt"
    assert "safetensors" in lock_content, "safetensors missing from requirements.lock.txt"


def test_no_duplicate_suffixed_packages() -> None:
    """Verify no near-duplicate suffixed packages (e.g. httpcore2, httpx2, nest-asyncio2)."""
    lock_path = ROOT_DIR / "requirements.lock.txt"
    assert lock_path.exists(), f"requirements.lock.txt not found at {lock_path}"

    disallowed_packages = {"httpcore2", "httpx2", "nest-asyncio2"}
    content = lock_path.read_text(encoding="utf-8")

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = line.split("==")[0].split(">=")[0].strip()
        assert pkg_name not in disallowed_packages, (
            f"Found disallowed suffixed package in requirements.lock.txt: {pkg_name}"
        )


def test_manifest_in_references_exist() -> None:
    """Verify that every path included or grafted in MANIFEST.in exists on disk."""
    manifest_path = ROOT_DIR / "MANIFEST.in"
    assert manifest_path.exists(), f"MANIFEST.in not found at {manifest_path}"

    content = manifest_path.read_text(encoding="utf-8")
    for line_idx, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        cmd = tokens[0]
        if cmd == "include":
            for path_pattern in tokens[1:]:
                # Expand simple file paths without wildcards
                if "*" not in path_pattern and "?" not in path_pattern:
                    target = ROOT_DIR / path_pattern
                    assert target.exists(), (
                        f"MANIFEST.in line {line_idx} references non-existent path: {path_pattern}"
                    )
        elif cmd == "graft":
            for dir_pattern in tokens[1:]:
                if "*" not in dir_pattern and "?" not in dir_pattern:
                    target = ROOT_DIR / dir_pattern
                    assert target.exists(), (
                        f"MANIFEST.in line {line_idx} grafts non-existent directory: {dir_pattern}"
                    )
