#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Generate and verify README registry tables for attention and compression families."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def build_family_table(registry: Any) -> tuple[str, dict[str, list[str]]]:
    """Build a markdown table (Family | Count | Members) from registry canonical keys."""
    registry._load_all()
    family_members: dict[str, list[str]] = defaultdict(list)
    for k in sorted(registry.canonical_keys()):
        obj = registry._dict[k]
        fam = getattr(obj, "FAMILY", "uncategorized") or "uncategorized"
        family_members[fam].append(k)

    lines = [
        "| Family | Count | Members |",
        "| :--- | :---: | :--- |",
    ]
    for fam in sorted(family_members.keys()):
        members_str = ", ".join(f"`{m}`" for m in sorted(family_members[fam]))
        lines.append(f"| {fam} | {len(family_members[fam])} | {members_str} |")

    return "\n".join(lines), family_members


def generate_updated_readme(readme_text: str) -> str:
    """Generate updated README text with honest counts and family tables."""
    attn_table, attn_families = build_family_table(ATTENTION_REGISTRY)
    comp_table, comp_families = build_family_table(COMPRESSOR_REGISTRY)

    total_attn = len(ATTENTION_REGISTRY.canonical_keys())
    num_attn_fam = len(attn_families)
    total_comp = len(COMPRESSOR_REGISTRY.canonical_keys())
    num_comp_fam = len(comp_families)

    # 1. Update line 768-775 (Section 5)
    readme_text = re.sub(
        r"## 5\. Registry and Hardware Scope \(\d+ Attention \+ \d+ Compression\)",
        f"## 5. Registry and Hardware Scope ({total_attn} Attention + {total_comp} Compression)",
        readme_text,
    )
    readme_text = re.sub(
        r"NeuralOps standardizes \*\*\d+.*?attention mechanisms across \d+ architectural families\*\* and \*\*\d+.*?compression engines across \d+ algorithmic paradigms\*\*",
        f"NeuralOps standardizes **{total_attn} distinct attention mechanisms across {num_attn_fam} architectural families** and **{total_comp} post-training compression engines across {num_comp_fam} algorithmic paradigms**",
        readme_text,
    )
    readme_text = re.sub(
        r"\|\s*\*\*Attention Modules \(`ATTENTION_REGISTRY`\)\*\*\s*\|\s*\*\*\d+\*\*\s*\|\s*\d+ foundational families",
        f"| **Attention Modules (`ATTENTION_REGISTRY`)** | **{total_attn}** | {num_attn_fam} foundational families",
        readme_text,
    )
    readme_text = re.sub(
        r"\|\s*\*\*Compression Engines \(`COMPRESSOR_REGISTRY`\)\*\*\s*\|\s*\*\*\d+\*\*\s*\|\s*\d+ algorithmic paradigms",
        f"| **Compression Engines (`COMPRESSOR_REGISTRY`)** | **{total_comp}** | {num_comp_fam} algorithmic paradigms",
        readme_text,
    )

    # 2. Update Section 13
    readme_text = re.sub(
        r"## 13\. Attention Zoo \(\d+ Canonical Modules\)",
        f"## 13. Attention Zoo ({total_attn} Canonical Modules)",
        readme_text,
    )
    readme_text = re.sub(
        r"### Mathematical Foundations & Architectural Taxonomy \(\d+ Families\)",
        f"### Mathematical Foundations & Architectural Taxonomy ({num_attn_fam} Families)",
        readme_text,
    )
    readme_text = re.sub(
        r"NeuralOps categorizes all \d+ attention mechanisms across \*\*\d+ distinct mathematical paradigms\*\*",
        f"NeuralOps categorizes all {total_attn} attention mechanisms across **{num_attn_fam} distinct mathematical paradigms**",
        readme_text,
    )

    # Replace taxonomy table in §13 (either text box or old markdown table)
    sec13_pattern = re.compile(
        r"(NeuralOps categorizes all \d+ attention mechanisms across \*\*\d+ distinct mathematical paradigms\*\*.*?\n\n)"
        r"(?:```text\n┌─.*?└─.*?```|\| Family \| Count \| Members \|.*?(?=\n\n(?:####|\#\#\#)))",
        re.DOTALL,
    )
    readme_text = sec13_pattern.sub(f"\\1{attn_table}", readme_text, count=1)

    # Update audit table heading in §13
    readme_text = re.sub(
        r"### Complete \d+-Module Forensic Hardware Audit Table",
        f"### Complete {total_attn}-Module Forensic Hardware Audit Table",
        readme_text,
    )

    # Filter alias rows from audit table in §13 and renumber
    audit_sec_pattern = re.compile(
        r"(### Complete \d+-Module Forensic Hardware Audit Table\n\n"
        r"\| # \| Attention Module Name \| Status \| FP16 \| BF16 \| FP32 \| Bwd Pass \| Latency \(ms\) \| Peak VRAM \| Notes / Error \|\n"
        r"\| :-: \| :--- \| :-: \| :-: \| :-: \| :-: \| :-: \| :-: \| :-: \| :--- \|\n)"
        r"(.*?)(?=\n\n\*For complete mathematical references)",
        re.DOTALL,
    )
    m = audit_sec_pattern.search(readme_text)
    if m:
        prefix = m.group(1)
        body = m.group(2)
        canonical_keys_set = set(ATTENTION_REGISTRY.canonical_keys())
        filtered_rows = []
        for line in body.strip().split("\n"):
            parts = [p.strip() for p in line.split("|")]
            # parts: ['', '#', '`module_name`', 'Status', ...]
            if len(parts) >= 4:
                mod_name = parts[2].replace("`", "")
                if mod_name in canonical_keys_set:
                    filtered_rows.append(parts)

        # Sort and renumber
        filtered_rows.sort(key=lambda p: p[2].replace("`", ""))
        new_lines = []
        for idx, parts in enumerate(filtered_rows, 1):
            parts[1] = str(idx)
            new_lines.append("| " + " | ".join(parts[1:-1]) + " |")

        new_body = "\n".join(new_lines)
        readme_text = readme_text[: m.start()] + prefix + new_body + readme_text[m.end() :]

    # 3. Update Section 14
    readme_text = re.sub(
        r"## 14\. Compression Engines \(\d+ Canonical Methods\)",
        f"## 14. Compression Engines ({total_comp} Canonical Methods)",
        readme_text,
    )
    readme_text = re.sub(
        r"### Mathematical Taxonomy & Reduction Scope \(\d+ Canonical Paradigms\)",
        f"### Mathematical Taxonomy & Reduction Scope ({num_comp_fam} Canonical Paradigms)",
        readme_text,
    )
    readme_text = re.sub(
        r"NeuralOps unifies \d+ post-training model compression engines across \*\*\d+ distinct mathematical paradigms\*\*",
        f"NeuralOps unifies {total_comp} post-training model compression engines across **{num_comp_fam} distinct mathematical paradigms**",
        readme_text,
    )

    sec14_pattern = re.compile(
        r"(NeuralOps unifies \d+ post-training model compression engines across \*\*\d+ distinct mathematical paradigms\*\*.*?\n\n)"
        r"(?:```text\n┌─.*?└─.*?```|\| Family \| Count \| Members \|.*?(?=\n\n(?:####|\#\#\#)))",
        re.DOTALL,
    )
    readme_text = sec14_pattern.sub(f"\\1{comp_table}", readme_text, count=1)

    return readme_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", default="README.md", help="Path to README.md")
    parser.add_argument("--check", action="store_true", help="Check if README is up to date")
    args = parser.parse_args()

    readme_path = Path(args.readme)
    if not readme_path.exists():
        print(f"Error: {readme_path} not found", file=sys.stderr)
        return 1

    current_text = readme_path.read_text(encoding="utf-8")
    updated_text = generate_updated_readme(current_text)

    if args.check:
        if current_text != updated_text:
            print(
                f"Error: {readme_path} is not up to date with registry counts and tables.",
                file=sys.stderr,
            )
            return 1
        print(f"{readme_path} is up to date with registry tables and counts.")
        return 0

    readme_path.write_text(updated_text, encoding="utf-8")
    print(f"Updated {readme_path} with generated registry tables and counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
