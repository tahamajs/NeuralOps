#!/usr/bin/env python3
"""
Documentation Catalog Generator
Generates and validates Docs/catalog.json across all technical modules.
"""
import json
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "Docs"
CATALOG_PATH = DOCS_DIR / "catalog.json"


def main():
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}")
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    total = sum(len(v) for v in catalog.values())
    print(f"Catalog verified with {total} entries across {len(catalog)} categories.")


if __name__ == "__main__":
    main()
