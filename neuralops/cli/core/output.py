#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Output Stream Separation and Pure JSON Emitter for NeuralOps CLI.

Discipline Contract:
  - stdout: Strictly clean benchmark tables or raw JSON documents.
  - stderr: Diagnostic telemetry, progress notifications, and warnings.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    console = Console()
    err_console = Console(stderr=True)
except ImportError:
    HAS_RICH = False
    console = None
    err_console = None


def emit(
    payload: Any,
    *,
    json_mode: bool = False,
    quiet: bool = False,
    err: bool = False,
) -> None:
    """Emits output to either stdout (results) or stderr (telemetry/errors)."""
    dest = sys.stderr if err else sys.stdout

    if json_mode:
        if not err:
            if isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                    print(json.dumps(parsed, indent=2, sort_keys=True), file=dest)
                except Exception:
                    print(payload, file=dest)
            else:
                print(json.dumps(payload, indent=2, sort_keys=True), file=dest)
        return

    if quiet and not err:
        return

    if HAS_RICH and (console is not None if not err else err_console is not None):
        target_console = err_console if err else console
        if target_console is not None:
            target_console.print(payload)
            return

    print(payload, file=dest)


def emit_ci_marker(marker_type: str, **kwargs: Any) -> None:
    """Emits a standardized machine-parseable progress line to stderr for CI loggers."""
    kvs = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"NEURALOPS_{marker_type.upper()} {kvs}", file=sys.stderr)


class CliReporter:
    """Standardized output manager maintaining stdout/stderr stream discipline."""

    def __init__(self, quiet: bool = False, verbose: bool = False, json_mode: bool = False) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.json_mode = json_mode

    def print_json(self, data: Any) -> None:
        """Print structured JSON to stdout."""
        print(json.dumps(data, indent=2, default=str), file=sys.stdout)

    def log(self, msg: str) -> None:
        """Print progress or status to stderr unless quiet."""
        if not self.quiet and not self.json_mode:
            print(msg, file=sys.stderr)

    def print_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        """Print rich or ASCII table to stdout."""
        if self.json_mode:
            records = [dict(zip(headers, row)) for row in rows]
            self.print_json(records)
            return

        if HAS_RICH and console is not None:
            table = Table(title=title, header_style="bold cyan")
            for h in headers:
                table.add_column(h)
            for r in rows:
                table.add_row(*r)
            console.print(table)
        else:
            print(f"=== {title} ===")
            print(" | ".join(headers))
            print("-" * 50)
            for r in rows:
                print(" | ".join(r))
