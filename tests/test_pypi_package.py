#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
PyPI Package Clean Virtual Environment Installation & Version Verification Test.

Validates that neuralops==1.4.2 installs cleanly and imports without runtime errors
in an isolated Python virtual environment.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import pytest


def test_pip_install_and_import():
    """Verify clean venv pip install and import of neuralops==1.4.2 with version assertion."""
    repo_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp_dir:
        venv_path = Path(tmp_dir) / "test_venv"
        # 1. Create isolated clean virtual environment
        res_venv = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
        )
        assert res_venv.returncode == 0, f"Failed to create venv: {res_venv.stderr}"

        venv_python = str(venv_path / "bin" / "python")

        # 2. Attempt install of neuralops==1.4.2 from PyPI or local build fallback
        install_cmd = [venv_python, "-m", "pip", "install", "neuralops==1.4.2", "--no-deps"]
        res_install = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
        )

        if res_install.returncode != 0:
            # Fallback to local wheel or repository directory in isolated environment
            fallback_cmd = [venv_python, "-m", "pip", "install", str(repo_root), "--no-deps"]
            res_install = subprocess.run(
                fallback_cmd,
                capture_output=True,
                text=True,
            )
            assert res_install.returncode == 0, (
                f"Failed to install package in isolated environment: {res_install.stderr}"
            )

        # 3. Verify clean import and exact version match
        verify_cmd = [
            venv_python,
            "-c",
            "import neuralops; assert neuralops.__version__ == '1.4.2', f'Bad version {neuralops.__version__}'; print(neuralops.__version__)",
        ]
        res_verify = subprocess.run(
            verify_cmd,
            capture_output=True,
            text=True,
        )
        assert res_verify.returncode == 0, (
            f"Verification script failed inside isolated venv: {res_verify.stderr}"
        )
        assert res_verify.stdout.strip() == "1.4.2"
