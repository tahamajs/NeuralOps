#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Reporting and structured console output formatters for NeuralOps examples.
"""

from __future__ import annotations

from typing import Any, Sequence


def section(title: str, width: int = 70) -> None:
    """Print a prominent boxed section header."""
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def kv(key: str, value: Any, width: int = 28) -> None:
    """Print a key-value pair in aligned format."""
    print(f"  {key.ljust(width)}: {value}")


def table(rows: Sequence[Sequence[Any]], headers: Sequence[str] | None = None) -> None:
    """Print a simple aligned table."""
    if not rows and not headers:
        return

    all_data = [list(map(str, headers))] + [[str(c) for c in r] for r in rows] if headers else [[str(c) for c in r] for r in rows]
    col_widths = [max(len(row[col_idx]) for row in all_data) for col_idx in range(len(all_data[0]))]

    if headers:
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "-+-".join("-" * w for w in col_widths)
        print(f"  {header_line}")
        print(f"  {sep_line}")

    for row in rows:
        row_str = " | ".join(str(val).ljust(w) for val, w in zip(row, col_widths))
        print(f"  {row_str}")


def pass_fail(label: str, passed: bool, details: str | None = None) -> None:
    """Print a standardized PASS/FAIL status line."""
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"  {tag} {label}"
    if details:
        msg += f" ({details})"
    print(msg)


def skip(label: str, reason: str) -> None:
    """Print a standardized SKIPPED status line."""
    print(f"  [SKIPPED] {label}: {reason}")
