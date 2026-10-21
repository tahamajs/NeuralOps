#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verify that the package root stays importable without PyTorch."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_import_neuralops_without_torch() -> None:
    """Allow root import but explain how to install PyTorch on API access."""
    script = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class BlockTorch(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ModuleNotFoundError(
                        "No module named 'torch'", name=fullname
                    )
                return None

        sys.meta_path.insert(0, BlockTorch())

        import neuralops
        print(f"version={neuralops.__version__}")

        try:
            _ = neuralops.replace_attention
        except ImportError as exc:
            message = str(exc)
            if "neuralops[cpu]" not in message:
                print(f"ERROR: unclear message: {message}")
                sys.exit(1)
            print("GUARD_OK")
        else:
            print("ERROR: torch-dependent API loaded without torch")
            sys.exit(1)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        f"Subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "version=" in result.stdout
    assert "GUARD_OK" in result.stdout
