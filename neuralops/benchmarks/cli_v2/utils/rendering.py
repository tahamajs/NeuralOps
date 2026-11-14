"""Rendering utilities for NeuralOps CLI v2.

Provides terminal formatting, error card rendering, tables, and status badges
with graceful fallback when Rich is not installed.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
    _CONSOLE: Optional[Console] = Console()
    _ERR_CONSOLE: Optional[Console] = Console(stderr=True)
except ImportError:  # allowed-swallow: rich is an optional terminal UI enhancement
    HAS_RICH = False
    _CONSOLE = None
    _ERR_CONSOLE = None


def get_console() -> Optional[Console]:
    """Get the primary stdout Rich console if available.

    Returns:
        Rich Console instance or None.
    """
    return _CONSOLE


def get_err_console() -> Optional[Console]:
    """Get the primary stderr Rich console if available.

    Returns:
        Rich Console instance or None.
    """
    return _ERR_CONSOLE


def print_cli_error(
    msg: str,
    hint: Optional[str] = None,
    title: str = "Execution Error",
) -> None:
    """Print a formatted CLI error card with optional actionable suggestion.

    Args:
        msg: Primary error message text.
        hint: Optional suggestion or troubleshooting hint.
        title: Title of the error panel or banner.
    """
    if HAS_RICH and _ERR_CONSOLE is not None:
        content = f"[bold red]✗ {msg}[/]"
        if hint:
            content += f"\n\n[dim cyan]Suggestion:[/] {hint}"
        panel = Panel(
            content,
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            padding=(0, 2),
        )
        _ERR_CONSOLE.print(panel)
    else:
        print(f"✗ Error: {msg}", file=sys.stderr)
        if hint:
            print(f"  Suggestion: {hint}", file=sys.stderr)


def render_badge(status: str) -> str:
    """Render a colored or formatted badge for a given status string.

    Args:
        status: The status text (e.g. PASS, FAIL, SKIP, RUNNING).

    Returns:
        Formatted string suitable for console display.
    """
    status_upper = status.upper()
    if HAS_RICH:
        color_map = {
            "PASS": "green",
            "SUCCESS": "green",
            "OK": "green",
            "FAIL": "red",
            "FAILED": "red",
            "ERROR": "red",
            "SKIP": "yellow",
            "SKIPPED": "yellow",
            "WARN": "yellow",
            "WARNING": "yellow",
            "RUNNING": "cyan",
            "PENDING": "blue",
        }
        color = color_map.get(status_upper, "white")
        return f"[bold {color}][{status_upper}][/bold {color}]"
    return f"[{status_upper}]"


def render_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    header_style: str = "bold cyan",
) -> None:
    """Render a structured table to stdout.

    Args:
        title: Title of the table.
        headers: Column header strings.
        rows: Rows of string data matching the header length.
        header_style: Rich style for headers (ignored without Rich).
    """
    if HAS_RICH and _CONSOLE is not None:
        table = Table(title=title, header_style=header_style)
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*row)
        _CONSOLE.print(table)
    else:
        print(f"=== {title} ===")
        header_line = " | ".join(headers)
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(row))
        print()


def render_summary_card(
    title: str,
    fields: Sequence[tuple[str, str]],
) -> None:
    """Render a key-value summary card.

    Args:
        title: Title of the summary section.
        fields: List of (Key, Value) pairs.
    """
    if HAS_RICH and _CONSOLE is not None:
        table = Table(title=title, header_style="bold cyan", show_header=False)
        table.add_column("Key", style="bold")
        table.add_column("Value", style="green")
        for key, val in fields:
            table.add_row(key, val)
        _CONSOLE.print(table)
    else:
        print("═" * 53)
        print(f"{title}:")
        for key, val in fields:
            print(f"  {key:<20} {val}")
        print("═" * 53)
