#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying CUDA JIT bindings isolation and environment sanitation."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.utils.cpp_extension as cpp_ext


def test_no_global_monkey_patch_on_import() -> None:
    """Verify that importing _bindings does not mutate torch.utils.cpp_extension globally."""
    orig_load_inline = cpp_ext.load_inline
    orig_load = cpp_ext.load

    import neuralops.ops.cuda._bindings as bindings_mod

    assert cpp_ext.load_inline is orig_load_inline
    assert cpp_ext.load is orig_load
    assert callable(bindings_mod.load_inline)
    assert callable(bindings_mod.load)


def test_get_cuda_arch_flags_empty_without_cuda() -> None:
    """Verify that get_cuda_arch_flags returns empty list when CUDA is unavailable."""
    from neuralops.ops.cuda._bindings import get_cuda_arch_flags

    if not torch.cuda.is_available():
        assert get_cuda_arch_flags() == []


def test_no_hardcoded_hpc_paths() -> None:
    """Verify that no hardcoded cluster paths remain in _bindings.py."""
    bindings_file = Path(__file__).resolve().parent.parent / "neuralops" / "ops" / "cuda" / "_bindings.py"
    text = bindings_file.read_text(encoding="utf-8")
    assert "/share/apps" not in text
    assert "~/bin" not in text
