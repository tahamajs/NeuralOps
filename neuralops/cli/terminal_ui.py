#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Terminal UI (Re-export facade over neuralops.benchmarks.terminal_ui).
"""

from __future__ import annotations

from neuralops.benchmarks.terminal_ui import (
    BANNER_ART,
    COMMAND_GROUPS,
    HAS_RICH,
    __version__,
    console,
    create_progress,
    format_table,
    get_theme,
    print_banner,
    print_command_group_help,
    print_error,
    print_gpu_telemetry_bar,
    print_info,
    print_markdown,
    print_panel,
    print_status_card,
    print_success,
    print_table,
    print_warning,
    prompt_confirm,
    prompt_select,
    run_interactive_ui,
    run_wizard,
    search_modules,
)
from neuralops.benchmarks.terminal_ui import *  # noqa: F401, F403

__all__ = [
    "HAS_RICH",
    "BANNER_ART",
    "COMMAND_GROUPS",
    "console",
    "create_progress",
    "format_table",
    "get_theme",
    "print_banner",
    "print_command_group_help",
    "print_error",
    "print_gpu_telemetry_bar",
    "print_info",
    "print_markdown",
    "print_panel",
    "print_status_card",
    "print_success",
    "print_table",
    "print_warning",
    "prompt_confirm",
    "prompt_select",
    "run_interactive_ui",
    "run_wizard",
    "search_modules",
    "__version__",
]
