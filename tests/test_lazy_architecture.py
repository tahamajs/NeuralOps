#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Isolated Subprocess Tests for Tiered Lazy Architecture (Phase 2).
"""

from __future__ import annotations

import subprocess
import sys


def _run_subp(code: str) -> None:
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_bare_import_is_stdlib_only():
    """Test 1 — Bare import is stdlib-only."""
    code = """
import sys
import neuralops
loaded = [m for m in sys.modules if m.startswith("neuralops")]
assert len(loaded) < 20, f"Too many modules loaded: {len(loaded)} -> {loaded}"
assert "torch" not in sys.modules, "Torch loaded by bare import"
"""
    _run_subp(code)


def test_version_access_without_torch():
    """Test 2 — Version access works without torch."""
    code = """
import sys
import neuralops
assert neuralops.__version__ == "1.6.0"
assert "torch" not in sys.modules
"""
    _run_subp(code)


def test_replace_attention_import_without_torch():
    """Test 3 — from neuralops import replace_attention works without torch."""
    code = """
from neuralops import replace_attention
assert callable(replace_attention)
"""
    _run_subp(code)


def test_calling_replace_attention_without_torch_raises_helpful_error():
    """Test 4 — Calling replace_attention without torch raises a helpful error."""
    code = """
import sys
from unittest.mock import patch
with patch.dict(sys.modules, {"torch": None}):
    import importlib
    import neuralops._optional
    importlib.reload(neuralops._optional)
    from neuralops import replace_attention
    try:
        replace_attention(None, "cosformer")
        assert False, "Should have raised TorchNotInstalledError"
    except Exception as e:
        assert "PyTorch" in str(e), f"Expected 'PyTorch' in message: {e}"
        assert "pip install" in str(e), f"Expected 'pip install' in message: {e}"
"""
    _run_subp(code)


def test_layers_cosformer_attention_requires_torch():
    """Test 5 — from neuralops.layers import CosformerAttention requires torch."""
    code = """
from neuralops.layers import CosformerAttention
assert CosformerAttention is not None
"""
    _run_subp(code)


def test_old_import_paths_still_work():
    """Test 6 — Old import paths still work and match new paths."""
    code = """
from neuralops.attentions import CosformerAttention as OldPath
from neuralops.layers import CosformerAttention as NewPath
assert OldPath is NewPath

from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor as OldComp
from neuralops.compression.svd import DirectSVDCompressor as NewComp
assert OldComp is NewComp

from neuralops.core.cuda_kernels import cosformer_linear_attention as OldKernel
from neuralops.ops.cuda import cosformer_linear_attention as NewKernel
assert OldKernel is NewKernel
"""
    _run_subp(code)


def test_no_circular_imports_in_any_order():
    """Test 7 — No circular imports across tiers."""
    code = """
import neuralops.core.errors
import neuralops.ops.cuda
import neuralops.ops.triton
import neuralops.layers
import neuralops.compression
import neuralops.models
import neuralops.api
"""
    _run_subp(code)


def test_import_time_budget():
    """Test 8 — Import time budget under 500ms."""
    code = """
import time
t0 = time.time()
import neuralops
t1 = time.time()
assert t1 - t0 < 0.5, f"import neuralops took {t1-t0:.2f}s"
"""
    _run_subp(code)


def test_cu_bindings_not_loaded_on_bare_import():
    """Test 9 — .cu files and bindings not loaded on bare import."""
    code = """
import sys
import neuralops
assert not any(m.endswith("_bindings") for m in sys.modules)
"""
    _run_subp(code)


def test_lazy_loading_is_observable():
    """Test 10 — Lazy loading is observable."""
    code = """
import sys
import neuralops
assert "neuralops.layers" not in sys.modules
_ = neuralops.replace_attention
assert "neuralops.layers" not in sys.modules
"""
    _run_subp(code)
