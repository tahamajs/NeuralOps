#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 2 Migration & Tier Architecture Verification Tests (P002-P005).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_collection_modifyitems(items):
    """Ensure pure filesystem audit tests in this module are not skipped on CPU hosts."""
    for item in items:
        # Remove skip markers added by conftest for cuda/triton keywords
        item.keywords.pop("gpu", None)
        item.keywords.pop("triton", None)
        item.own_markers = [m for m in item.own_markers if m.name not in ("gpu", "triton", "skip")]


def test_layers_populated():
    """Verify neuralops/layers has at least 100 python files."""
    layers_py = list((REPO_ROOT / "neuralops" / "layers").rglob("*.py"))
    assert len(layers_py) >= 100, f"Expected >= 100, found {len(layers_py)}"


def test_ops_cuda_populated():
    """Verify neuralops/ops/cuda has at least 100 entries."""
    cuda_items = list((REPO_ROOT / "neuralops" / "ops" / "cuda").iterdir())
    assert len(cuda_items) >= 100, f"Expected >= 100, found {len(cuda_items)}"


def test_ops_triton_populated():
    """Verify neuralops/ops/triton has at least 25 python files."""
    triton_py = list((REPO_ROOT / "neuralops" / "ops" / "triton").glob("*.py"))
    assert len(triton_py) >= 25, f"Expected >= 25, found {len(triton_py)}"


def test_compression_populated():
    """Verify neuralops/compression has at least 70 python files."""
    comp_py = list((REPO_ROOT / "neuralops" / "compression").rglob("*.py"))
    assert len(comp_py) >= 70, f"Expected >= 70, found {len(comp_py)}"


def test_old_path_works():
    """Verify ATTENTION_REGISTRY from old path is identical to new path."""
    from neuralops.attentions import ATTENTION_REGISTRY
    from neuralops.layers import ATTENTION_REGISTRY as NEW
    assert ATTENTION_REGISTRY is NEW


def test_new_path_works():
    """Verify from neuralops.layers import CosformerAttention succeeds."""
    from neuralops.layers import CosformerAttention
    assert CosformerAttention is not None


def test_no_real_files_in_old_paths():
    """Verify real implementation files (>500 bytes) in neuralops/attentions/ is <= 10."""
    attn_dir = REPO_ROOT / "neuralops" / "attentions"
    large_files = [
        f for f in attn_dir.rglob("*.py")
        if f.stat().st_size > 500
    ]
    assert len(large_files) <= 10, f"Too many large files in old attentions path: {len(large_files)} -> {large_files}"


def test_no_circular_imports():
    """Verify isolated imports of new tier paths succeed without circular dependencies."""
    code = """
import neuralops.layers
import neuralops.ops.cuda
import neuralops.ops.triton
import neuralops.compression
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Import failed: {res.stderr}"
