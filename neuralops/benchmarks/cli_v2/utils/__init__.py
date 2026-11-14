"""CLI v2 utility modules."""

from __future__ import annotations

from neuralops.benchmarks.cli_v2.utils.completion import (
    CANONICAL_COMMANDS,
    SUPPORTED_SHELLS,
    generate_completion_script,
)
from neuralops.benchmarks.cli_v2.utils.rendering import (
    HAS_RICH,
    get_console,
    get_err_console,
    print_cli_error,
    render_badge,
    render_summary_card,
    render_table,
)

__all__ = [
    "CANONICAL_COMMANDS",
    "HAS_RICH",
    "SUPPORTED_SHELLS",
    "generate_completion_script",
    "get_console",
    "get_err_console",
    "print_cli_error",
    "render_badge",
    "render_summary_card",
    "render_table",
]
