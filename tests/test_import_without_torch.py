#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying torch-free import and clean exception raising in subprocesses."""

from __future__ import annotations

import subprocess
import sys


def test_import_neuralops_without_torch() -> None:
    """Verify that importing neuralops succeeds in an environment where torch is not installed."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import neuralops\n"
        "assert neuralops.__version__ == '2.0.6'\n"
        "print('SUCCESS')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Importing neuralops without torch failed:\n{res.stdout}\n{res.stderr}"
    assert "SUCCESS" in res.stdout


def test_import_api_without_torch_fails_with_clear_error() -> None:
    """Verify that importing neuralops.api without torch raises a clean MissingDependencyError."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "try:\n"
        "    import neuralops.api\n"
        "except Exception as e:\n"
        "    assert 'NeuralOps requires PyTorch' in str(e)\n"
        "    assert 'pip install \"neuralops[core]\"' in str(e)\n"
        "    print('CLEAN_ERROR_RAISED')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Importing neuralops.api without torch did not raise expected error:\n{res.stdout}\n{res.stderr}"
    assert "CLEAN_ERROR_RAISED" in res.stdout
