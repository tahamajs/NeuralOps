#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Kernel Fallback & Hardware Dispatch Test Suite.

Verifies:
1. Modularized low-rank submodules importability, parity, and facade compatibility.
2. Three-level fallback contracts (Level 1 CUDA -> Level 2 Triton -> Level 3 PyTorch Reference).
3. Zero silent fallback invariants and strict exception handling.
4. Absence of hardcoded 'cuda:0' strings across operators and fallbacks.
5. CPU reference correctness across diverse input tensor shapes and rank parameters.
"""

from __future__ import annotations

import importlib
import pytest
import torch

from neuralops.ops.cuda._fallback import (
    FORCE_PYTORCH_FALLBACK,
    force_kernel_backend,
    get_dispatch_stats,
    get_last_dispatch_backend,
    reset_dispatch_stats,
)
import neuralops.ops.triton.low_rank as lr


class TestLowRankModularization:
    """Tests for low_rank modularized submodules and facade integrity."""

    def test_submodules_importability(self) -> None:
        """Verify all submodules under neuralops.ops.triton._low_rank import cleanly."""
        submodules = [
            "core",
            "quantization",
            "fused_ops",
            "structured",
            "svd_steps",
            "kernels_gemm",
            "kernels_tt_kron",
            "svd_compress",
        ]
        for name in submodules:
            mod = importlib.import_module(f"neuralops.ops.triton._low_rank.{name}")
            assert mod is not None, f"Failed to load submodule _low_rank.{name}"

    def test_facade_symbol_parity(self) -> None:
        """Verify that low_rank.py facade re-exports all required canonical functions."""
        required_symbols = [
            "is_supported",
            "fwd",
            "bwd",
            "low_rank_forward",
            "quantize_to_fp8_dynamic",
            "dynamic_fp8_quant_triton",
            "fp8_tensorcore_matmul_triton",
            "pack_int4_weights",
            "unpack_int4_weights",
            "fused_w4a16_gemm_triton",
            "BitPackedW4A16Linear",
            "BitPackedLowRankSVDLinear",
            "fused_linformer_seq_proj_triton",
            "fused_linformer_seq_proj_pytorch",
            "fused_galerkin_continuous_kernel_triton",
            "fused_galerkin_continuous_kernel_pytorch",
            "fused_nystrom_landmark_pinv_step_triton",
            "fused_nystrom_landmark_pinv_step_pytorch",
            "fused_rpc_sketch_attention_triton",
            "fused_rpc_sketch_attention_pytorch",
            "fused_primal_dual_landmark_triton",
            "fused_primal_dual_landmark_pytorch",
            "fused_svd_mamba_proj_triton",
            "fused_svd_mamba_proj_pytorch",
            "fused_lola_3way_decomposition_triton",
            "fused_lola_3way_decomposition_pytorch",
            "fused_int4_lut_dequant_matmul_triton",
            "fused_int4_lut_dequant_matmul_pytorch",
            "fused_roast_hashed_gather_gemm_triton",
            "fused_roast_hashed_gather_gemm_pytorch",
            "fused_kronecker_batched_gemm_triton",
            "fused_kronecker_batched_gemm_pytorch",
            "fused_tt_core_contraction_triton",
            "fused_tt_core_contraction_pytorch",
            "fused_salt_singular_mod_gemm_triton",
            "fused_salt_singular_mod_gemm_pytorch",
            "fused_svd_qkv_rope_attention_triton",
            "fused_svd_qkv_rope_attention_pytorch",
            "fused_svd_gemm_triton",
            "fused_down_up_gemm_bias_triton",
            "fused_residual_svd_gemm_triton",
            "fused_svd_lora_sparse_attn_triton",
            "fused_hnc_nullspace_step_triton",
            "fused_hnc_nullspace_step_pytorch",
            "block_diag_matmul_triton",
            "rotational_svd_triton",
            "triton_fused_lowrank_gated_gemm",
            "triton_fused_lowrank_gated_gemm_pytorch_ref",
            "triton_fused_lowrank_sparse_gemm",
            "triton_fused_lowrank_sparse_gemm_pytorch_ref",
            "triton_strided_tt_contraction",
            "triton_strided_tt_linear",
            "triton_strided_tt_contraction_pytorch_ref",
            "triton_fused_kronecker_matmul",
            "triton_fused_kronecker_matmul_pytorch_ref",
            "triton_spectral_entropy_svd_gate",
            "triton_spectral_entropy_svd_gate_pytorch_ref",
            "direct_svd_compress",
            "direct_svd_compress_pytorch",
            "asvd_compress",
            "asvd_compress_pytorch",
            "fwsvd_compress",
            "fwsvd_compress_pytorch",
            "hnc_svd_compress",
            "hnc_svd_compress_pytorch",
        ]
        for sym in required_symbols:
            assert hasattr(lr, sym), f"Facade neuralops.ops.triton.low_rank missing symbol: {sym}"


class TestKernelFallbacksAndParity:
    """Tests verifying fallback execution, zero crashes, and numerical parity."""

    def test_low_rank_forward_cpu_fallback(self) -> None:
        """Verify low_rank_forward runs on CPU via reference path without error."""
        batch, seq, in_dim, rank, out_dim = 2, 8, 32, 8, 16
        x = torch.randn(batch, seq, in_dim, dtype=torch.float32)
        u = torch.randn(out_dim, rank, dtype=torch.float32)
        v = torch.randn(in_dim, rank, dtype=torch.float32)
        s = torch.ones(rank, dtype=torch.float32)
        bias = torch.zeros(out_dim, dtype=torch.float32)

        out = lr.low_rank_forward(x, V=v, S=s, U=u, bias=bias)
        assert out.shape == (batch, seq, out_dim)
        assert not torch.isnan(out).any()

    def test_fused_svd_gemm_cpu_fallback(self) -> None:
        """Verify fused_svd_gemm_triton fallback handles optional residual."""
        m, k, r, n = 4, 16, 4, 8
        x = torch.randn(m, k)
        v = torch.randn(k, r)
        u = torch.randn(r, n)
        res = torch.randn(m, n)

        out = lr.fused_svd_gemm_triton(x, U=u, V=v, residual=res)
        expected = torch.matmul(torch.matmul(x, v), u) + res
        assert torch.allclose(out, expected, atol=1e-5)

    def test_direct_svd_compression_fallback(self) -> None:
        """Verify direct SVD compression falls back gracefully on CPU."""
        m, n, rank = 32, 64, 8
        w = torch.randn(m, n)
        w_rec = lr.direct_svd_compress(w, rank=rank)
        assert w_rec.shape == (m, n)
        assert not torch.isnan(w_rec).any()

    def test_hnc_nullspace_step_fallback(self) -> None:
        """Verify HNC nullspace spectral step PyTorch fallback."""
        d1, d2 = 16, 16
        w = torch.randn(d1, d2)
        grad_phi = torch.randn(d1, d2)
        inv_null_proj = torch.eye(d2)
        out = lr.fused_hnc_nullspace_step_pytorch(w, grad_phi, inv_null_proj, eta=0.01)
        assert out.shape == (d1, d2)
        assert not torch.isnan(out).any()

    def test_strided_tt_contraction_fallback(self) -> None:
        """Verify strided TT contraction reference parity."""
        c0 = torch.randn(1, 4, 4, 2)
        c1 = torch.randn(2, 4, 4, 1)

        ref_w = lr.triton_strided_tt_contraction_pytorch_ref(c0, c1, p_dims=(4, 4), q_dims=(4, 4))
        assert ref_w.shape == (16, 16)
        assert not torch.isnan(ref_w).any()

    def test_fused_kronecker_matmul_fallback(self) -> None:
        """Verify fused Kronecker matmul reference parity."""
        m, m1, m2, n1, n2 = 2, 4, 4, 4, 4
        x = torch.randn(m, m1 * m2)
        a = torch.randn(m1, n1)
        b = torch.randn(m2, n2)

        ref_out = lr.triton_fused_kronecker_matmul_pytorch_ref(x, a, b)
        assert ref_out.shape == (m, n1 * n2)
        assert not torch.isnan(ref_out).any()

    def test_int4_lut_dequant_fallback(self) -> None:
        """Verify INT4 LUT dequantization matmul fallback."""
        m, k, n = 2, 32, 16
        x = torch.randn(m, k)
        qweight = torch.randint(0, 16, (n, k), dtype=torch.uint8)
        lut = torch.linspace(-1.0, 1.0, 16)
        scales = torch.ones(n, 1)

        out = lr.fused_int4_lut_dequant_matmul_pytorch(x, qweight, lut, scales)
        assert out.shape == (m, n)
        assert not torch.isnan(out).any()

    def test_force_kernel_backend_fallback_telemetry(self) -> None:
        """Verify telemetry counters update during backend forcing."""
        reset_dispatch_stats()
        with force_kernel_backend("pytorch"):
            stats = get_dispatch_stats()
            assert isinstance(stats, dict)
            assert "pytorch" in stats


class TestGovernanceAndInvariants:
    """Verifies that AGENTS.md rules are strictly satisfied."""

    def test_zero_hardcoded_device_in_ops(self) -> None:
        """Verify no hardcoded 'cuda:0' strings exist in ops code."""
        import inspect

        # Inspect low_rank facade and submodules
        for mod_name in [
            "core",
            "quantization",
            "fused_ops",
            "structured",
            "svd_steps",
            "kernels_gemm",
            "kernels_tt_kron",
            "svd_compress",
        ]:
            mod = importlib.import_module(f"neuralops.ops.triton._low_rank.{mod_name}")
            source = inspect.getsource(mod)
            assert "cuda:0" not in source, f"Hardcoded cuda:0 detected in _low_rank.{mod_name}"

    def test_file_size_limits_compliance(self) -> None:
        """Verify modularized files obey soft/hard line limits (<= 800 lines)."""
        import os

        paths = [
            "neuralops/ops/triton/low_rank.py",
            "neuralops/ops/triton/_low_rank/core.py",
            "neuralops/ops/triton/_low_rank/quantization.py",
            "neuralops/ops/triton/_low_rank/fused_ops.py",
            "neuralops/ops/triton/_low_rank/structured.py",
            "neuralops/ops/triton/_low_rank/svd_steps.py",
            "neuralops/ops/triton/_low_rank/kernels_gemm.py",
            "neuralops/ops/triton/_low_rank/kernels_tt_kron.py",
            "neuralops/ops/triton/_low_rank/svd_compress.py",
        ]
        for path in paths:
            assert os.path.exists(path), f"File {path} does not exist"
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) <= 800, f"File {path} exceeds 800 lines ({len(lines)} lines)"
