#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive PEP 8 Style and Indentation Normalizer for NeuralOps.

"""

from __future__ import annotations

import ast
import re
from pathlib import Path

EXCLUDE_PARTS = {".git", "__pycache__", "build", "dist", ".venv", "venv", "archive", "_archive", ".ipynb_checkpoints"}


def format_python_source(source: str) -> str:
    # 1. Expand tabs to 4 spaces
    source = source.expandtabs(4)

    # 2. Process line by line for trailing whitespace and single-line statements
    lines = source.splitlines()
    new_lines: list[str] = []

    in_docstring = False
    docstring_char = ""

    for line in lines:
        stripped = line.rstrip()

        # Track multi-line docstrings so we don't alter text inside docstrings
        for quote in ('"""', "'''"):
            cnt = line.count(quote)
            if not in_docstring and cnt % 2 == 1:
                in_docstring = True
                docstring_char = quote
                break
            elif in_docstring and quote == docstring_char and cnt % 2 == 1:
                in_docstring = False
                docstring_char = ""
                break

        if in_docstring:
            new_lines.append(stripped)
            continue

        # Single-line def expansion: def name(args) -> type: return expr
        m_def = re.match(r"^([ \t]*def\s+[a-zA-Z0-9_]+\([^)]*\)\s*(?:->\s*[^:]+)?:\s*)([a-zA-Z0-9_].+)$", stripped)
        if m_def:
            prefix = m_def.group(1).rstrip()
            body = m_def.group(2).strip()
            indent = len(prefix) - len(prefix.lstrip())
            new_lines.append(prefix)
            new_lines.append(" " * (indent + 4) + body)
            continue

        # Single-line if expansion: if cond: stmt (not part of comprehension or multi-line)
        m_if = re.match(r"^([ \t]*if\s+.+?:\s*)([a-zA-Z0-9_].+)$", stripped)
        if m_if and not stripped.endswith(":") and not stripped.startswith("#"):
            prefix = m_if.group(1).rstrip()
            body = m_if.group(2).strip()
            # Ensure it's not a quote continuation
            if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
                indent = len(prefix) - len(prefix.lstrip())
                new_lines.append(prefix)
                new_lines.append(" " * (indent + 4) + body)
                continue

        new_lines.append(stripped)

    # 3. Normalize excess consecutive blank lines (max 2 consecutive)
    result_lines: list[str] = []
    blank_run = 0
    for l in new_lines:
        if not l:
            blank_run += 1
            if blank_run <= 2:
                result_lines.append(l)
        else:
            blank_run = 0
            result_lines.append(l)

    # 4. Ensure single trailing newline
    return "\n".join(result_lines).rstrip() + "\n"


def main() -> None:
    root = Path(".")
    target_dirs = ["neuralops", "scripts", "benchmarks", "evaluation", "tests"]
    formatted_count = 0

    for d in target_dirs:
        dir_path = root / d
        if not dir_path.is_dir():
            continue

        for p in sorted(dir_path.rglob("*.py")):
            if any(part in p.parts for part in EXCLUDE_PARTS) or not p.is_file():
                continue

            try:
                orig = p.read_text(encoding="utf-8")
            except Exception:
                continue

            formatted = format_python_source(orig)
            if formatted != orig:
                try:
                    ast.parse(formatted, filename=str(p))
                    p.write_text(formatted, encoding="utf-8")
                    formatted_count += 1
                except Exception as exc:
                    print(f"Skipping {p} due to format error: {exc}")

    print(f"Successfully normalized PEP 8 style and indentation across {formatted_count} Python files.")


if __name__ == "__main__":
    main()
