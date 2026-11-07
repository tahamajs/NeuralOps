#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for scripts/ci/check_skip_ratio.py."""

from pathlib import Path
import pytest

from scripts.ci.check_skip_ratio import evaluate_junit_xml


def _write_junit_xml(path: Path, tests: int, failures: int, errors: int, skipped: int) -> None:
    content = f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="{tests}" errors="{errors}" failures="{failures}" skipped="{skipped}" time="1.234">
  </testsuite>
</testsuites>
"""
    path.write_text(content)


def test_skip_ratio_passes_on_healthy_run(tmp_path: Path):
    """Verify check passes when tests pass and skip ratio is low."""
    xml_path = tmp_path / "junit.xml"
    _write_junit_xml(xml_path, tests=100, failures=0, errors=0, skipped=10)

    ok, stats, errors = evaluate_junit_xml(xml_path, max_skip_ratio=0.50)
    assert ok is True
    assert stats["passed"] == 90
    assert stats["skip_ratio"] == 0.10
    assert len(errors) == 0


def test_skip_ratio_fails_when_all_skipped(tmp_path: Path):
    """Verify check fails when all tests are skipped (passed == 0)."""
    xml_path = tmp_path / "junit.xml"
    _write_junit_xml(xml_path, tests=50, failures=0, errors=0, skipped=50)

    ok, stats, errors = evaluate_junit_xml(xml_path, max_skip_ratio=0.50)
    assert ok is False
    assert any("Zero tests passed" in e for e in errors)


def test_skip_ratio_fails_when_exceeding_threshold(tmp_path: Path):
    """Verify check fails when skip ratio exceeds max_skip_ratio threshold."""
    xml_path = tmp_path / "junit.xml"
    _write_junit_xml(xml_path, tests=100, failures=0, errors=0, skipped=60)

    ok, stats, errors = evaluate_junit_xml(xml_path, max_skip_ratio=0.50)
    assert ok is False
    assert any("exceeds maximum permitted threshold" in e for e in errors)


def test_skip_ratio_fails_on_test_failures(tmp_path: Path):
    """Verify check fails when failures are present."""
    xml_path = tmp_path / "junit.xml"
    _write_junit_xml(xml_path, tests=100, failures=2, errors=0, skipped=10)

    ok, stats, errors = evaluate_junit_xml(xml_path, max_skip_ratio=0.50)
    assert ok is False
    assert any("test(s) failed" in e for e in errors)
