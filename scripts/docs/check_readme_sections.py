#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Checks that section numbers in root README.md have no duplicates.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys


def check_readme_sections(readme_path: Path) -> int:
    content = readme_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    section_re = re.compile(r"^(#{2,4})\s+([0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$")
    sections: list[tuple[int, str, str]] = []

    in_code_block = False

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        m = section_re.match(line)
        if m:
            level = len(m.group(1))
            sec_num = m.group(2)
            title = m.group(3).strip()
            sections.append((idx, f"{level}:{sec_num}", title))

    counts = Counter(s[1] for s in sections)
    duplicates = {num: count for num, count in counts.items() if count > 1}

    if duplicates:
        print(f"Error: Found duplicate numbered sections in {readme_path}:", file=sys.stderr)
        for idx, sec_tag, title in sections:
            if sec_tag in duplicates:
                print(f"  Line {idx}: [{sec_tag}] {title}", file=sys.stderr)
        return 1

    print(f"README section numbering clean ({len(sections)} numbered sections verified).")
    return 0


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent.parent
    readme_path = root_dir / "README.md"
    if not readme_path.exists():
        print(f"README.md not found at {readme_path}", file=sys.stderr)
        return 1
    return check_readme_sections(readme_path)


if __name__ == "__main__":
    sys.exit(main())
