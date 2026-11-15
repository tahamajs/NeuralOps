#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for HFHubPusher and SECRET_PATTERNS security scanning.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from neuralops.benchmarks.results.hf_sync import HFHubPusher, SECRET_PATTERNS


def test_secret_patterns_count():
    """Verify SECRET_PATTERNS contains at least 8 patterns."""
    assert len(SECRET_PATTERNS) >= 8


def test_scan_for_secrets(tmp_path: Path):
    """Verify scan_for_secrets catches known credentials and passes clean files."""
    pusher = HFHubPusher(repo_id="test/repo", token="dummy_token")

    clean_file = tmp_path / "clean.json"
    clean_file.write_text('{"accuracy": 0.95, "model": "bert"}', encoding="utf-8")
    assert pusher.scan_for_secrets(clean_file) is True

    # Test each secret pattern
    test_secrets = [
        "hf_1234567890123456789012345678901234",
        "sk-1234567890123456789012345",
        "ghp_123456789012345678901234567890123456",
        "ghs_123456789012345678901234567890123456",
        "AKIAIOSFODNN7EXAMPLE",
        "AIzaSyD-1234567890123456789012345678901",
        "pypi-AgEIcHlwaS5vcmc12345678901234567890123456789012345678901234567890",
        'api_key = "12345678901234567890"',
    ]

    for idx, secret in enumerate(test_secrets):
        dirty_file = tmp_path / f"dirty_{idx}.txt"
        dirty_file.write_text(f"Here is a key: {secret}", encoding="utf-8")
        assert pusher.scan_for_secrets(dirty_file) is False, f"Failed to detect pattern: {secret}"


def test_generate_hub_readme():
    """Verify generate_hub_readme produces valid Markdown."""
    pusher = HFHubPusher()
    readme = pusher.generate_hub_readme("neuralops/benchmarks-zoo")
    assert "license: apache-2.0" in readme
    assert "neuralops/benchmarks-zoo" in readme
