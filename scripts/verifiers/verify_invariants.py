#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Authoritative Invariant Verification Utility

Audits repository invariants according to AGENTS.md §6:
1. Attention Registry Keys (Historical: 131, Audited Canonical: 115)
2. Compressor Registry Keys (Historical: 72, Canonical v2.1.0 Zoo: 82)
3. Triton Inventory Units (Canonical: 58)
4. CUDA Package Units (Canonical: 140 packages, 141 with __pycache__)
5. Canonical Notebooks (Canonical: 94)

Enforces AGENTS.md §0.1 verification honesty and fail-closed reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def audit_invariants() -> dict[str, dict]:
    """Execute physical counts across filesystem and component registries."""
    # 1. Attentions & Compressors
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        ATTENTION_REGISTRY._load_all()
        COMPRESSOR_REGISTRY._load_all()

        attn_canonical = len(ATTENTION_REGISTRY.canonical_keys())
        attn_distinct = len(ATTENTION_REGISTRY.distinct_classes())
        comp_canonical = len(COMPRESSOR_REGISTRY.canonical_keys())
        comp_distinct = len(COMPRESSOR_REGISTRY.distinct_classes())
    except Exception as e:
        attn_canonical = -1
        attn_distinct = -1
        comp_canonical = -1
        comp_distinct = -1

    # 2. Triton inventory units
    triton_dir = ROOT_DIR / "neuralops" / "ops" / "triton"
    triton_files = sorted([p.name for p in triton_dir.glob("*.py") if p.name != "__init__.py"])
    triton_count = len(triton_files)

    # 3. CUDA packages
    cuda_dir = ROOT_DIR / "neuralops" / "ops" / "cuda"
    cuda_pkgs = sorted([
        p.name for p in cuda_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    ])
    cuda_count = len(cuda_pkgs)
    cuda_count_with_pycache = len([
        p.name for p in cuda_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])

    # 4. Notebooks
    notebooks_dir = ROOT_DIR / "notebooks"
    notebooks = sorted([p.name for p in notebooks_dir.glob("*.ipynb")])
    notebook_count = len(notebooks)

    results = {
        "attentions": {
            "canonical_keys": attn_canonical,
            "distinct_classes": attn_distinct,
            "documented_historical": 131,
            "status": "PASS" if attn_canonical in (115, 131) else "DRIFT",
            "note": "Audited canonical classes = 115 (excluding template and experimental layers per test_registry_counts.py)"
        },
        "compressors": {
            "canonical_keys": comp_canonical,
            "distinct_classes": comp_distinct,
            "documented_historical": 72,
            "documented_v210_zoo": 82,
            "status": "PASS" if comp_canonical in (72, 82) else "DRIFT",
            "note": "v2.1.0 82-Compressor Zoo expanded from baseline 72 (AGENTS.md §21.3)"
        },
        "triton_modules": {
            "file_count": triton_count,
            "expected": 58,
            "status": "PASS" if triton_count == 58 else "DRIFT",
            "note": "Exact match with AGENTS.md §6 canonical Triton inventory"
        },
        "cuda_packages": {
            "package_count": cuda_count,
            "dir_count_with_pycache": cuda_count_with_pycache,
            "expected_packages": 140,
            "expected_with_pycache": 141,
            "status": "PASS" if cuda_count in (140, 141) else "DRIFT",
            "note": "140 standalone CUDA kernel packages (141 when __pycache__ directory is present)"
        },
        "notebooks": {
            "notebook_count": notebook_count,
            "expected": 94,
            "status": "PASS" if notebook_count == 94 else "DRIFT",
            "note": "Exact match with AGENTS.md §6 canonical notebook inventory"
        }
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Authoritative Invariant Verification Gatekeeper")
    parser.add_argument("--check", action="store_true", help="Exit with non-zero code on invariant drift")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    results = audit_invariants()

    all_pass = all(item["status"] == "PASS" for item in results.values())

    if args.json:
        payload = {
            "overall_status": "PASS" if all_pass else "FAIL",
            "invariants": results,
        }
        print(json.dumps(payload, indent=2))
        return 0 if (all_pass or not args.check) else 1

    print("================================================================================")
    print("                    NEURALOPS AUTHORITATIVE INVARIANTS AUDIT                    ")
    print("================================================================================")
    for name, data in results.items():
        status = data["status"]
        status_str = f"[\033[92m{status}\033[0m]" if status == "PASS" else f"[\033[91m{status}\033[0m]"
        print(f"{status_str} {name.upper()}:")
        for k, v in data.items():
            if k != "status":
                print(f"    - {k}: {v}")
    print("================================================================================")
    print(f"OVERALL STATUS: {'PASS' if all_pass else 'FAIL'}")
    print("================================================================================")

    if args.check and not all_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
