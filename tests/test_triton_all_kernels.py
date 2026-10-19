#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and registry test suite for all Triton kernels in neuralops.

Coverage:
- Import integrity of all Triton kernel modules and fallbacks
- Compatibility verification on both CPU and CUDA
- End-to-end integration across attention architectures utilizing Triton kernels
"""
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels import (
    HAS_TRITON,
    fms_scan,
    fms_scan_triton,
    gdn_scan,
    gdn_scan_triton,
    gla_scan,
    gla_scan_triton,
    gqa,
    gqa_pytorch_fallback,
    hgrn2_scan,
    hgrn2_scan_triton,
    linear_attn,
    linear_attn_pytorch_fallback,
    low_rank,
    low_rank_pytorch_fallback,
    selective_scan_fn,
    softplus,
    sparse_attn,
    sparse_attn_pytorch_fallback,
    state_update,
    state_update_pytorch_fallback,
    triton_dispatch,
)
from neuralops.attentions.linear.triton_cosformer import TritonCosformerAttention
from neuralops.attentions.svd.triton_svd_linear_attention import TritonSVDLinearAttention


def test_kernel_exports_and_imports():
    """Verify all kernels and fallbacks are exported and accessible."""
    assert callable(linear_attn_pytorch_fallback)
    assert callable(sparse_attn_pytorch_fallback)
    assert callable(low_rank_pytorch_fallback)
    assert callable(state_update_pytorch_fallback)
    assert callable(gqa_pytorch_fallback)
    assert callable(softplus)
    assert callable(triton_dispatch)

    assert linear_attn is not None
    assert sparse_attn is not None
    assert low_rank is not None
    assert state_update is not None
    assert gqa is not None
    assert fms_scan is not None
    assert gdn_scan is not None
    assert gla_scan is not None
    assert hgrn2_scan is not None


def test_cpu_fallbacks_execution():
    """Verify fallbacks execute reliably on CPU without crashing or NaN."""
    x = torch.randn(4, 4)
    from neuralops.core.triton_kernels.softplus import softplus_fn
    assert torch.allclose(softplus_fn(x), F.softplus(x))
    assert callable(softplus)

    u = torch.randn(4, 8)
    v = torch.randn(8, 4)
    act = low_rank_pytorch_fallback(x, u, v)
    assert act.shape == (4, 4)
    assert torch.isfinite(act).all()

    k = torch.randn(2, 2, 8, 16)
    v_t = torch.randn(2, 2, 8, 16)
    k_exp, v_exp = gqa_pytorch_fallback(k, v_t, num_kv_heads=2, num_heads=4)
    assert k_exp.shape == (2, 4, 8, 16)
    assert v_exp.shape == (2, 4, 8, 16)

    q = torch.randn(2, 2, 16, 16)
    k_l = torch.randn(2, 2, 16, 16)
    v_l = torch.randn(2, 2, 16, 16)
    out_l = linear_attn_pytorch_fallback(q, k_l, v_l)
    assert out_l.shape == (2, 2, 16, 16)
    assert torch.isfinite(out_l).all()

    u_s = torch.randn(2, 4, 8)
    delta = torch.randn(2, 4, 8).sigmoid() * 0.1
    A = -torch.rand(8, 4)
    B = torch.randn(2, 4, 4)
    C = torch.randn(2, 4, 4)
    out_s, state = state_update_pytorch_fallback(u_s, delta, A, B, C)
    assert out_s.shape == (2, 4, 8)
    assert state.shape == (2, 8, 4)
    assert torch.isfinite(out_s).all()


def test_cosformer_and_svd_linear_attention_modules():
    """Verify higher-level attention layers built on Triton kernels instantiate and run."""
    x = torch.randn(2, 8, 64)

    cosformer = TritonCosformerAttention(embed_dim=64, num_heads=4)
    out_cos, _ = cosformer(x)
    assert out_cos.shape == (2, 8, 64)
    assert torch.isfinite(out_cos).all()

    svd_attn = TritonSVDLinearAttention(embed_dim=64, num_heads=4, rank=16)
    out_svd, _ = svd_attn(x)
    assert out_svd.shape == (2, 8, 64)
    assert torch.isfinite(out_svd).all()


def test_triton_kernels_unification_and_interoperability():
    """Verify bidirectional accessibility between neuralops.kernels.triton and neuralops.core.triton_kernels."""
    import neuralops
    import neuralops.kernels.triton as nkt
    import neuralops.core.triton_kernels as ctk

    flagship_callables = [
        "triton_fused_lowrank_gated_gemm",
        "triton_fused_lowrank_sparse_gemm",
        "triton_strided_tt_contraction",
        "triton_strided_tt_linear",
        "triton_fused_hadamard_w4a16_gemm",
        "triton_lut_codebook_dequant_gemm",
        "triton_fused_kronecker_matmul",
        "triton_chunked_delta_gate_scan",
        "triton_cosformer_linear_attention",
        "triton_block_sparse_streaming_attn",
        "triton_hgrn2_fused_bwd_scan",
        "triton_spectral_entropy_svd_gate",
        "triton_fused_act_scale_quant_gemm",
        "triton_fused_latent_kv_cache_attention",
    ]
    for fn_name in flagship_callables:
        assert hasattr(nkt, fn_name), f"Missing {fn_name} in neuralops.kernels.triton"
        assert callable(getattr(nkt, fn_name))
        assert hasattr(ctk, fn_name), f"Missing {fn_name} in neuralops.core.triton_kernels"
        assert callable(getattr(ctk, fn_name))

    py_refs = [
        "triton_fused_lowrank_gated_gemm_pytorch_ref",
        "triton_fused_lowrank_sparse_gemm_pytorch_ref",
        "triton_strided_tt_contraction_pytorch_ref",
        "triton_fused_kronecker_matmul_pytorch_ref",
        "triton_spectral_entropy_svd_gate_pytorch_ref",
        "triton_fused_hadamard_w4a16_gemm_pytorch_ref",
        "triton_lut_codebook_dequant_gemm_pytorch_ref",
        "triton_fused_act_scale_quant_gemm_pytorch_ref",
        "triton_chunked_delta_gate_scan_pytorch_ref",
        "triton_cosformer_linear_attention_pytorch_ref",
        "triton_block_sparse_streaming_attn_pytorch_ref",
        "triton_hgrn2_fused_bwd_scan_pytorch_ref",
        "triton_fused_latent_kv_cache_attention_pytorch_ref",
    ]
    for ref_name in py_refs:
        assert hasattr(nkt, ref_name), f"Missing {ref_name} in neuralops.kernels.triton"
        assert callable(getattr(nkt, ref_name))
        assert hasattr(ctk, ref_name), f"Missing {ref_name} in neuralops.core.triton_kernels"
        assert callable(getattr(ctk, ref_name))

    assert hasattr(neuralops.kernels, "triton")
    assert neuralops.kernels.triton is nkt

    from neuralops.core.triton_kernels import chunked_delta_gate_scan as cdgs_alias
    from neuralops.kernels.triton import chunked_delta_gate_scan as cdgs_direct
    assert cdgs_alias is cdgs_direct

    from neuralops.kernels.triton import gla_scan as gla_from_nkt
    assert gla_from_nkt is ctk.gla_scan
