#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying single source of truth for terminal_ui (F-55).
"""

from __future__ import annotations

from pathlib import Path

import neuralops
import neuralops.benchmarks.terminal_ui as bench_ui
import neuralops.cli.terminal_ui as cli_ui


def test_terminal_ui_single_implementation():
    """Verify cli/terminal_ui.py is a thin re-export facade under 100 lines."""
    cli_path = Path(cli_ui.__file__).resolve()
    content = cli_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    assert len(lines) < 100, f"cli/terminal_ui.py is too long ({len(lines)} lines); must be a thin facade"
    assert "from neuralops.benchmarks.terminal_ui import" in content, (
        "cli/terminal_ui.py must re-export from neuralops.benchmarks.terminal_ui"
    )


def test_terminal_ui_version_from_package():
    """Verify terminal_ui.__version__ matches the authoritative package version."""
    assert bench_ui.__version__ == neuralops.__version__
    assert cli_ui.__version__ == neuralops.__version__
