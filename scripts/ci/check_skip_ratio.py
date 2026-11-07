#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CI Skip Ratio Guard.

Parses pytest JUnit XML results and verifies that CPU smoke tests actually executed
rather than being silently skipped. Fails if passed == 0 or if the skip ratio
exceeds the permitted threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def evaluate_junit_xml(
    xml_path: Path, max_skip_ratio: float = 0.50
) -> tuple[bool, dict[str, int | float], list[str]]:
    """Parse JUnit XML file and evaluate skip ratio and execution criteria."""
    if not xml_path.is_file():
        return False, {}, [f"JUnit XML report not found at {xml_path}"]

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        return False, {}, [f"Failed to parse JUnit XML: {exc}"]

    # Handle either <testsuite> or <testsuites> top element
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0

    for suite in suites:
        total_tests += int(suite.attrib.get("tests", 0))
        total_failures += int(suite.attrib.get("failures", 0))
        total_errors += int(suite.attrib.get("errors", 0))
        total_skipped += int(suite.attrib.get("skipped", 0))

    passed = total_tests - (total_failures + total_errors + total_skipped)
    skip_ratio = (total_skipped / total_tests) if total_tests > 0 else 0.0

    stats = {
        "tests": total_tests,
        "passed": passed,
        "failures": total_failures,
        "errors": total_errors,
        "skipped": total_skipped,
        "skip_ratio": skip_ratio,
        "max_skip_ratio": max_skip_ratio,
    }

    errors: list[str] = []
    if total_tests == 0:
        errors.append("No tests collected in JUnit report.")
    elif passed <= 0:
        errors.append(f"Zero tests passed out of {total_tests} collected.")

    if total_failures > 0:
        errors.append(f"{total_failures} test(s) failed.")
    if total_errors > 0:
        errors.append(f"{total_errors} test collection/execution error(s).")

    if skip_ratio > max_skip_ratio:
        errors.append(
            f"Skip ratio {skip_ratio:.1%} exceeds maximum permitted threshold {max_skip_ratio:.1%} "
            f"({total_skipped}/{total_tests} skipped)."
        )

    return len(errors) == 0, stats, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check pytest skip ratio from JUnit XML.")
    parser.add_argument("junit_xml", type=Path, help="Path to pytest JUnit XML report")
    parser.add_argument(
        "--max-skip-ratio",
        type=float,
        default=0.50,
        help="Maximum allowed ratio of skipped tests (default: 0.50 = 50%%)",
    )
    args = parser.parse_args(argv)

    ok, stats, errors = evaluate_junit_xml(args.junit_xml, max_skip_ratio=args.max_skip_ratio)

    print(
        f"[REPORT] Tests: {stats.get('tests', 0)}, Passed: {stats.get('passed', 0)}, "
        f"Skipped: {stats.get('skipped', 0)} ({stats.get('skip_ratio', 0.0):.1%}), "
        f"Failures: {stats.get('failures', 0)}, Errors: {stats.get('errors', 0)}"
    )

    if ok:
        print("[SUCCESS] Test execution and skip ratio are within acceptable bounds.")
        return 0

    print("[FAILURE] Skip ratio check failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
