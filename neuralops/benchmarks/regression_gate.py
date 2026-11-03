#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Fail-closed benchmark regression comparison for persisted result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark result not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("metrics"), dict):
        raise ValueError(f"Invalid benchmark result schema: {path}")
    return data


def compare_results(baseline_path: Path, candidate_path: Path, metric: str, max_regression: float) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if metric not in baseline["metrics"] or metric not in candidate["metrics"]:
        raise KeyError(f"Metric {metric!r} is missing from baseline or candidate")
    baseline_value = baseline["metrics"][metric]
    candidate_value = candidate["metrics"][metric]
    if not isinstance(baseline_value, (int, float)) or not isinstance(candidate_value, (int, float)):
        raise TypeError(f"Metric {metric!r} must be numeric in both results")
    regression = float(candidate_value) - float(baseline_value)
    passed = regression <= max_regression
    return {
        "status": "PASS" if passed else "FAIL_REGRESSION",
        "metric": metric,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "regression": regression,
        "max_regression": max_regression,
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--max-regression", type=float, required=True)
    args = parser.parse_args()
    result = compare_results(args.baseline, args.candidate, args.metric, args.max_regression)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
