#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying benchmark catalog integrity and registry population."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def test_catalog_points_to_existing_files():
    """Assert every benchmark entry in benchmarks/benchmark_catalog.json exists under neuralops/benchmarks/."""
    catalog_path = ROOT / "benchmarks" / "benchmark_catalog.json"
    assert catalog_path.exists(), "benchmark_catalog.json missing at repo root"

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks_dir = ROOT / "neuralops" / "benchmarks"
    categories = data.get("categories", {})

    missing: list[str] = []
    for cat_name, scripts in categories.items():
        for script in scripts:
            target = benchmarks_dir / script
            if not target.exists():
                missing.append(f"{cat_name}: {script} (expected at {target})")

    assert not missing, f"Catalog entries point to non-existent files: {missing}"


def test_registry_not_empty():
    """Assert BENCHMARK_REGISTRY is non-empty after import, refuting 0-entries claim."""
    from neuralops.benchmarks.registry import BENCHMARK_REGISTRY

    assert len(BENCHMARK_REGISTRY) > 0, "BENCHMARK_REGISTRY is unexpectedly empty (0 entries)"
    assert len(BENCHMARK_REGISTRY) >= 200, f"Expected >= 200 registered benchmark specs, got {len(BENCHMARK_REGISTRY)}"
