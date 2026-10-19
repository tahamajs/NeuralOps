#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Utility to audit, validate, format, and sync Hooshaai Hugging Face model cards."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REQUIRED_YAML_FIELDS = [
    "language",
    "license",
    "library_name",
    "tags",
    "pipeline_tag",
    "datasets",
    "metrics",
]

REQUIRED_SECTIONS = [
    "Model Details",
    "Architecture Overview",
    "Mathematical Formulation",
    "Empirical Benchmark Table",
    "Quickstart Code",
    "Hardware & Software Requirements",
    "Citation",
]


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Extract frontmatter and body from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if not match:
        return {}, content
    
    yaml_block = match.group(1)
    body = match.group(2)
    
    fields: dict[str, str] = {}
    current_key = None
    for line in yaml_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            parts = line.split(":", 1)
            current_key = parts[0].strip()
            val = parts[1].strip()
            fields[current_key] = val
        elif line.startswith("-") and current_key:
            val = line.lstrip("-").strip()
            if current_key not in fields or not isinstance(fields[current_key], list):
                fields[current_key] = []
            fields[current_key].append(val)
            
    return fields, body


def audit_model_card(card_path: Path, hub_org: str = "Hooshaai") -> list[str]:
    """Audit an individual model card markdown file."""
    errors: list[str] = []
    
    try:
        content = card_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"Unable to read file: {exc}"]

    fields, body = parse_frontmatter(content)
    
    if not fields:
        errors.append("Missing or invalid YAML frontmatter (delimited by '---')")
    else:
        for req in REQUIRED_YAML_FIELDS:
            if req not in fields:
                errors.append(f"Missing required YAML field: '{req}'")

    for section in REQUIRED_SECTIONS:
        if not re.search(rf"##\s+{re.escape(section)}", body, re.IGNORECASE):
            errors.append(f"Missing required markdown section: '## {section}'")

    expected_repo_prefix = f"{hub_org}/"
    if expected_repo_prefix not in body:
        errors.append(f"No reference found for organization repository '{expected_repo_prefix}*'")

    if "```python" not in body:
        errors.append("Quickstart python code block ('```python') is missing")

    if "@software" not in body and "@article" not in body and "@misc" not in body:
        errors.append("Citation BibTeX block is missing")

    return errors


def sync_cards(
    model_cards_dir: Path,
    export_dir: Path | None = None,
    hub_org: str = "Hooshaai",
    check_only: bool = False,
) -> int:
    """Audit and optionally export model cards."""
    card_files = sorted(
        p for p in model_cards_dir.glob("*.md")
        if p.name not in ("README.md", "INDEX.md")
    )
    
    if not card_files:
        print(f"Error: No model cards found in directory '{model_cards_dir}'")
        return 1

    print("=" * 72)
    print(f"Hooshaai Hugging Face Model Card Auditor & Sync Utility (Org: {hub_org})")
    print("=" * 72)
    print(f"Found {len(card_files)} model cards in: {model_cards_dir}\n")

    total_errors = 0
    valid_cards: list[tuple[Path, str]] = []

    for card in card_files:
        errs = audit_model_card(card, hub_org=hub_org)
        model_name = card.stem
        if errs:
            print(f"FAIL  {card.name}")
            for err in errs:
                print(f"      - {err}")
            total_errors += len(errs)
        else:
            print(f"PASS  {card.name} (Valid -> {hub_org}/{model_name})")
            valid_cards.append((card, model_name))

    print("\n" + "-" * 72)
    if total_errors > 0:
        print(f"Audit completed with {total_errors} error(s).")
        return 1

    print(f"All {len(card_files)} model cards passed strict schema validation!")

    if export_dir and not check_only:
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting Hub bundles to: {export_dir}")
        for card, model_name in valid_cards:
            target_repo_dir = export_dir / model_name
            target_repo_dir.mkdir(parents=True, exist_ok=True)
            readme_target = target_repo_dir / "README.md"
            readme_target.write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Exported: {readme_target.relative_to(export_dir.parent if export_dir.parent.exists() else export_dir)}")

        print(f"\nSuccessfully prepared {len(valid_cards)} Hugging Face Hub repositories.")

    print("=" * 72)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit, validate, and export Hooshaai Hugging Face Model Cards."
    )
    parser.add_argument(
        "--cards-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "docs" / "model_cards",
        help="Path to directory containing docs/model_cards/*.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run schema validation checks only without exporting.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Target directory to export Hugging Face Hub ready repositories.",
    )
    parser.add_argument(
        "--hub-org",
        type=str,
        default="Hooshaai",
        help="Target Hugging Face Hub organization identifier (default: Hooshaai).",
    )

    args = parser.parse_args()
    return sync_cards(
        model_cards_dir=args.cards_dir,
        export_dir=args.export_dir,
        hub_org=args.hub_org,
        check_only=args.check,
    )


if __name__ == "__main__":
    sys.exit(main())
