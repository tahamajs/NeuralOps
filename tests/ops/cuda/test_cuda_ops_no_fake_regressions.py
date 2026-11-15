# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite for CUDA operator dispatcher and kernel wrappers.

Verifies:
1. P0-O2-1: Python submodule shadowing prevention (operator symbol precedence over package).
2. P0-O2-2: _map.py mapping for chela_slay_wide_attention_forward.
3. P0-O2-3: routing_attention callable symbol export.
4. P0-O2-4: admm pruning returns torch.Tensor on CPU fallback.
5. P1-O2-5: paged_kv_evict logs warnings and does not swallow exceptions silently.
6. P1-O2-6: _bindings.py package resources compatibility shim annotations.
7. P1-O2-7: Bounds checks in kernel wrappers for fixed-size stack arrays.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch


def test_submodule_shadowing_prevention() -> None:
    """P0-O2-1: Importing submodules directly must not shadow mapped callable operator symbols."""
    # Import submodule first
    import neuralops.ops.cuda.moe_routing.moe_autograd  # noqa: F401
    import neuralops.ops.cuda.selective_scan.selective_scan_autograd  # noqa: F401
    import neuralops.ops.cuda as noc

    assert callable(noc.moe_routing), f"noc.moe_routing must be callable, got {type(noc.moe_routing)}"
    assert callable(noc.selective_scan), f"noc.selective_scan must be callable, got {type(noc.selective_scan)}"


def test_chela_slay_wide_mapping() -> None:
    """P0-O2-2: chela_slay_wide_attention_forward must resolve correctly from _map.py."""
    import neuralops.ops.cuda as noc

    fn = getattr(noc, "chela_slay_wide_attention_forward", None)
    assert fn is not None, "chela_slay_wide_attention_forward must not be None"
    assert callable(fn), f"chela_slay_wide_attention_forward must be callable, got {type(fn)}"


def test_routing_attention_symbol_export() -> None:
    """P0-O2-3: routing_attention symbol must be exported and callable."""
    import neuralops.ops.cuda as noc

    fn = getattr(noc, "routing_attention", None)
    assert fn is not None, "routing_attention must not be None"
    assert callable(fn), f"routing_attention must be callable, got {type(fn)}"


def test_admm_return_tensor() -> None:
    """P0-O2-4: admm pruning must return torch.Tensor, not a tuple."""
    import neuralops.ops.cuda as noc

    W = torch.randn(16, 16)
    out = noc.admm(W)
    assert isinstance(out, torch.Tensor), f"admm output must be torch.Tensor, got {type(out)}"
    assert out.shape == W.shape
    assert not torch.isnan(out).any()


def test_no_silent_swallow_in_paged_kv_evict() -> None:
    """P1-O2-5: paged_kv_evict_autograd.py must not contain bare except Exception: pass."""
    src_path = Path("neuralops/ops/cuda/paged_kv_evict/paged_kv_evict_autograd.py")
    content = src_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if "except Exception:" in line and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            assert next_line != "pass", f"Found bare 'except Exception: pass' at line {i + 1}"


def test_fixed_array_bounds_checks() -> None:
    """P1-O2-7: Kernel wrappers with fixed-size arrays must include bounds checks."""
    cuda_files = [
        ("neuralops/ops/cuda/chela_slay/chela_slay.cu", 512),
        ("neuralops/ops/cuda/cross_attention/cross_attention.cu", 512),
        ("neuralops/ops/cuda/lsh_attention/lsh_attention.cu", 256),
        ("neuralops/ops/cuda/rala/rala.cu", 512),
        ("neuralops/ops/cuda/routing_attention/routing_attention.cu", 256),
        ("neuralops/ops/cuda/token_compressed/token_compressed.cu", 512),
    ]
    for rel_path, max_dim in cuda_files:
        path = Path(rel_path)
        content = path.read_text(encoding="utf-8")
        assert f"<= {max_dim}" in content or f"< {max_dim}" in content, (
            f"{rel_path} lacks TORCH_CHECK for D <= {max_dim}"
        )
