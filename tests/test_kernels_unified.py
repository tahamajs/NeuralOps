# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Verification suite for Complete Unified Triton & CUDA Kernels in NeuralOps.

Validates:
1. Unified subpackage exports: `from neuralops.kernels import triton, cuda`
2. Complete Triton kernel inventory: `neuralops.kernels.list_triton_kernels()` (62 kernels)
3. Complete CUDA kernel inventory: `neuralops.kernels.list_cuda_kernels()` (114 kernels)
4. Bidirectional re-exports between `neuralops.kernels.triton` and `neuralops.core.triton_kernels`
5. Bidirectional re-exports between `neuralops.kernels.cuda` and `neuralops.core.cuda_kernels`
6. Category filtering for both Triton and CUDA kernel catalogs
"""

import pytest
import neuralops
import neuralops.kernels as nk


def test_kernels_subpackages_import():
    """Verify triton and cuda can be directly imported from neuralops.kernels."""
    from neuralops.kernels import triton, cuda

    assert triton is not None
    assert cuda is not None
    assert hasattr(nk, "triton")
    assert hasattr(nk, "cuda")


def test_triton_kernel_catalog_and_categories():
    """Verify Triton kernel catalog contains all 62 kernels and category filtering works."""
    all_triton = nk.list_triton_kernels()
    assert len(all_triton) == 62, f"Expected 62 Triton kernels, got {len(all_triton)}"

    standalone = nk.list_triton_kernels("standalone")
    assert len(standalone) == 13, f"Expected 13 standalone kernels, got {len(standalone)}"
    assert "cosformer_linear_attention" in standalone
    assert "fused_lowrank_gemm" in standalone
    assert "strided_tt" in standalone

    core = nk.list_triton_kernels("core")
    assert len(core) == 27, f"Expected 27 core categorized modules, got {len(core)}"
    assert "linear_attn" in core
    assert "cur_linear" in core
    assert "differential_attention" in core
    assert "dynamic_sparse_attention" in core
    assert "hydra_attention" in core
    assert "retnet_retention" in core

    root = nk.list_triton_kernels("root")
    assert len(root) == 2, f"Expected 2 root bridge modules, got {len(root)}"
    assert "triton_linear_attn" in root
    assert "triton_sparse_attn" in root

    extended = nk.list_triton_kernels("extended")
    assert len(extended) == 20, f"Expected 20 extended fused kernels, got {len(extended)}"
    assert "fused_swiglu" in extended
    assert "fused_rmsnorm" in extended


def test_cuda_kernel_catalog_and_categories():
    """Verify CUDA kernel catalog contains all 114 kernels and category filtering works."""
    all_cuda = nk.list_cuda_kernels()
    assert len(all_cuda) == 114, f"Expected 114 CUDA kernels, got {len(all_cuda)}"

    modules = nk.list_cuda_kernels("modules")
    assert len(modules) == 110, f"Expected 110 CUDA kernel modules, got {len(modules)}"
    assert "act_scale_gemm" in modules
    assert "cosformer" in modules
    assert "quantized_gemm" in modules

    operators = nk.list_cuda_kernels("operators")
    assert len(operators) >= 110, f"Expected >= 110 CUDA primary operators, got {len(operators)}"
    assert "fused_lowrank_gated_gemm" in operators
    assert "chunked_delta_gate_scan" in operators


def test_bidirectional_triton_compatibility():
    """Verify transparent bidirectional access between neuralops.kernels.triton and neuralops.core.triton_kernels."""
    import neuralops.kernels.triton as nkt
    import neuralops.core.triton_kernels as ctk

    assert hasattr(nkt, "cur_linear")
    assert hasattr(nkt, "differential_attention")
    assert hasattr(nkt, "hydra_attention")
    assert hasattr(nkt, "retnet_retention")

    assert hasattr(ctk, "cosformer_linear_attention")
    assert hasattr(ctk, "fused_lowrank_gemm")
    assert hasattr(ctk, "spectral_entropy_svd_gate")

    assert callable(nkt.list_triton_kernels)
    assert callable(ctk.list_triton_kernels)
    assert len(nkt.list_triton_kernels()) == len(ctk.list_triton_kernels()) == 62


def test_bidirectional_cuda_compatibility():
    """Verify transparent bidirectional access between neuralops.kernels.cuda and neuralops.core.cuda_kernels."""
    import neuralops.kernels.cuda as nkc
    import neuralops.core.cuda_kernels as cck

    assert hasattr(nkc, "fused_lowrank_gated_gemm")
    assert hasattr(cck, "fused_lowrank_gated_gemm")
    assert hasattr(nkc, "chunked_delta_gate_scan")
    assert hasattr(cck, "chunked_delta_gate_scan")

    assert hasattr(nkc, "act_scale_gemm")
    assert hasattr(cck, "act_scale_gemm")
    assert hasattr(nkc, "cosformer")
    assert hasattr(cck, "cosformer")

    assert callable(nkc.list_cuda_kernels)
    assert callable(cck.list_cuda_kernels)
    assert len(nkc.list_cuda_kernels()) == len(cck.list_cuda_kernels()) == 114


def test_root_neuralops_reexports():
    """Verify root neuralops package provides list_triton_kernels and list_cuda_kernels."""
    assert hasattr(neuralops, "list_triton_kernels")
    assert hasattr(neuralops, "list_cuda_kernels")
    assert callable(neuralops.list_triton_kernels)
    assert callable(neuralops.list_cuda_kernels)
    assert len(neuralops.list_triton_kernels()) == 62
    assert len(neuralops.list_cuda_kernels()) == 114
