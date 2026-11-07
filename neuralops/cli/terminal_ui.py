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

from neuralops.benchmarks.terminal_ui import *  # noqa: F401, F403
from neuralops.benchmarks.terminal_ui import (
    BANNER_ART,
    HAS_RICH,
    SUBCOMMAND_GROUPS,
    __version__,
    console,
    print_banner,
    print_hf_style_help,
    run_interactive_ui,
)

COMMAND_GROUPS = SUBCOMMAND_GROUPS

__all__ = [
    "HAS_RICH",
    "BANNER_ART",
    "COMMAND_GROUPS",
    "SUBCOMMAND_GROUPS",
    "console",
    "print_banner",
    "print_hf_style_help",
    "run_interactive_ui",
    "__version__",
]
