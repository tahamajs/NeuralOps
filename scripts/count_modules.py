#!/usr/bin/env python3
"""Single source of truth for NeuralOps module counts.

Walks ATTENTION_REGISTRY / COMPRESSOR_REGISTRY after auto-importing every
module file, deduplicates alias registrations by the underlying registered
object (the same semantic run_quality_sweep.py uses to build its sweep
universe), and reports raw vs canonical counts.

Never hand-type module counts into docs again -- regenerate from this script:
    .venv/bin/python scripts/count_modules.py
    .venv/bin/python scripts/count_modules.py --json   # machine-readable

It also reports any module file that failed to import (silent registration
gaps would otherwise make counts look lower than the file inventory).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = PROJECT_ROOT / "Codes"
for p in (str(CODES_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module  # noqa: E402

RESERVED_NAMES = {"full"}
PACKAGE_DIRS = {
    "attention": ("framework.attentions", PROJECT_ROOT / "Codes" / "framework" / "attentions"),
    "compressor": ("framework.compression_engine", PROJECT_ROOT / "Codes" / "framework" / "compression_engine"),
}


def import_all(package: str, directory: Path) -> list[str]:
    """Import every non-underscore .py in `directory`; return files that failed."""
    failures = []
    for file in sorted(directory.glob("*.py")):
        if file.stem.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package}.{file.stem}")
        except Exception as exc:  # noqa: BLE001 - report anything that breaks registration
            failures.append(f"{file.name}: {type(exc).__name__}: {exc}")
    return failures


def canonical_names(reg) -> dict[object, list[str]]:
    """Group registry keys by the object they resolve to (alias dedup)."""
    by_obj: dict[object, list[str]] = {}
    for name, obj in list(reg.items()):
        if name.lower() in RESERVED_NAMES or is_helper_module(name):
            continue
        by_obj.setdefault(obj, []).append(name)
    return by_obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    failures: dict[str, list[str]] = {}
    for label, (package, directory) in PACKAGE_DIRS.items():
        failures[label] = import_all(package, directory)

    att = canonical_names(ATTENTION_REGISTRY)
    comp = canonical_names(COMPRESSOR_REGISTRY)

    result = {
        "attention": {"raw_keys": len(ATTENTION_REGISTRY), "canonical": len(att)},
        "compressor": {"raw_keys": len(COMPRESSOR_REGISTRY), "canonical": len(comp)},
        "total_canonical": len(att) + len(comp),
        "import_failures": failures,
    }

    if args.json:
        result["attention_names"] = sorted(min(v) for v in att.values())
        result["compressor_names"] = sorted(min(v) for v in comp.values())
        print(json.dumps(result, indent=2))
    else:
        print("NeuralOps canonical module counts (alias-deduped, registry-resolved):")
        print(f"  attentions:   {result['attention']['canonical']} canonical  ({result['attention']['raw_keys']} raw registry keys)")
        print(f"  compressors:  {result['compressor']['canonical']} canonical  ({result['compressor']['raw_keys']} raw registry keys)")
        print(f"  total:        {result['total_canonical']}")
        for label, fails in failures.items():
            if fails:
                print(f"\nWARNING: {len(fails)} {label} module file(s) failed to import:")
                for f in fails:
                    print(f"  - {f}")

    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
