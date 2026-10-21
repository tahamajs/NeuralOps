#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Ensure family packages defer implementation imports until access."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_attention_family_import_is_lazy() -> None:
    """Importing one attention family loads only the requested implementation."""
    result = _run_isolated(
        """
        import sys
        import neuralops.attentions.linear
        assert "neuralops.attentions.linear.cosformer" not in sys.modules
        from neuralops.attentions.linear import CosformerAttention
        assert CosformerAttention.__module__ == "neuralops.attentions.linear.cosformer"
        loaded = [
            name for name in sys.modules
            if name.startswith("neuralops.attentions.linear.")
            and name != "neuralops.attentions.linear.__init__"
        ]
        assert loaded == ["neuralops.attentions.linear.cosformer"], loaded
        """
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_compressor_family_import_is_lazy() -> None:
    """Importing one compressor family defers its implementation imports."""
    result = _run_isolated(
        """
        import sys
        import neuralops.compression_engine.svd
        assert "neuralops.compression_engine.svd.residual_svd_compressor" not in sys.modules
        from neuralops.compression_engine.svd import ResSVDCompressor
        assert ResSVDCompressor.__module__ == (
            "neuralops.compression_engine.svd.residual_svd_compressor"
        )
        loaded = [
            name for name in sys.modules
            if name.startswith("neuralops.compression_engine.svd.")
            and name != "neuralops.compression_engine.svd.__init__"
        ]
        assert loaded == [
            "neuralops.compression_engine.svd.residual_svd_compressor"
        ], loaded
        """
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
