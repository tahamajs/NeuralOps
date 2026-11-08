#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying neuralops info command execution without torch installed."""

from __future__ import annotations

import subprocess
import sys


def test_info_without_torch() -> None:
    """Verify that neuralops info works without torch installed and displays 'torch: not installed'."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from neuralops.cli.info import cmd_info\n"
        "ret = cmd_info()\n"
        "assert ret == 0, f'cmd_info returned {ret}'\n"
        "print('INFO_WITHOUT_TORCH_SUCCESS')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Running cmd_info without torch failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "INFO_WITHOUT_TORCH_SUCCESS" in res.stdout
    combined = (res.stdout + res.stderr).lower()
    assert "not installed" in combined or "disabled" in combined
