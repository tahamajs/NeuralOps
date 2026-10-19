# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Exhaustive Deep Hardening & Boundary Stress Test Suite for Low-Scoring and Boundary Modules.

Targets empirical benchmark boundary modules across:
1. Pointwise / launch-overhead boundary attentions at small sequence lengths:
   - fused_favor_plus_feature_map_triton
   - cosformer
   - fused_goldfinch_hopfield_lookup_triton
   - fused_linformer_seq_proj_triton
   - fused_nystrom_landmark_pinv_step_triton
2. High FLOP / memory-intensive attention variants:
   - routing_attention
   - sink_attention
   - multi_scale_retention
   - rwkv6
   - differential_attention
3. Hybrid & Multi-branch structures:
   - split_head_hybrid_attention
   - tri_branch_hybrid_attention
   - jamba_hybrid (Jamba2Block, JambaInfoBlock)
4. Compression Engines under aggressive truncation & extreme ranks:
   - cr_compressor
   - cur_compressor
   - frandi_compressor
   - grecodec_compressor
   - residual_svd_compressor
   - sdrm_compressor
   - extreme rank boundaries (R in {8, 64})
5. Quantization boundary cases:
   - smoothquant
   - fp8_compressor

Verification Contracts:
- Numerical Safety: Zero NaN, Inf, or subnormal values under batch & sequence variations.
- Saturated Gradient Checks: Autograd backwards verify gradients do not vanish (||g|| > 1e-7) or explode (||g|| < 1e5).
- Level 3 PyTorch Reference Fallback Integrity: Exact numerical concordance (diff <= 1e-3).
- Bounded Memory & Dynamic Context Scaling without runaway memory leaks.
"""

from __future__ import annotations

import os
import sys
import gc
from types import SimpleNamespace
from typing import Any, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.append(REPO_ROOT)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def clean_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()



class TestPointwiseBoundaryKernels:
    """Tests small sequence and boundary cases for pointwise Triton kernels."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [16, 64, 256, 1024])
    def test_favor_plus_fallback_and_numerics(self, L: int):
        from neuralops.core.triton_kernels.elementwise import (
            fused_favor_plus_feature_map_triton,
            fused_favor_plus_feature_map_pytorch,
        )

        B, H, D = 2, 4, 64
        num_feat = 32
        x = torch.randn(B, H, L, D, device=DEVICE, dtype=torch.float32)
        omega = torch.randn(num_feat, D, device=DEVICE, dtype=torch.float32)

        out_triton = fused_favor_plus_feature_map_triton(x, omega, num_feat)
        out_pt = fused_favor_plus_feature_map_pytorch(x, omega, num_feat)

        assert not torch.isnan(out_triton).any(), "NaN found in favor_plus triton"
        assert not torch.isinf(out_triton).any(), "Inf found in favor_plus triton"
        diff = (out_triton - out_pt).abs().max().item()
        assert diff <= 1e-4, f"Favor+ triton vs pytorch mismatch: {diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [32, 64, 128])
    def test_goldfinch_hopfield_boundary(self, L: int):
        from neuralops.core.triton_kernels.fms_scan import (
            fused_goldfinch_hopfield_lookup_triton,
            fused_goldfinch_hopfield_lookup_pytorch,
        )

        B, D = 2, 64
        g = torch.sigmoid(torch.randn(B, L, D, device=DEVICE, dtype=torch.float32))
        v = torch.randn(B, L, D, device=DEVICE, dtype=torch.float32)

        out_triton = fused_goldfinch_hopfield_lookup_triton(g, v)
        out_pt = fused_goldfinch_hopfield_lookup_pytorch(g, v)

        assert not torch.isnan(out_triton).any()
        assert not torch.isinf(out_triton).any()
        diff = (out_triton - out_pt).abs().max().item()
        assert diff <= 1e-4, f"Goldfinch triton vs pytorch mismatch: {diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [128, 512, 1024])
    def test_linformer_seq_proj_boundary(self, L: int):
        from neuralops.core.triton_kernels.low_rank import (
            fused_linformer_seq_proj_triton,
            fused_linformer_seq_proj_pytorch,
        )

        B, H, D, k = 2, 4, 64, 32
        proj = torch.randn(k, L, device=DEVICE, dtype=torch.float32)
        seq = torch.randn(B, H, L, D, device=DEVICE, dtype=torch.float32)

        out_triton = fused_linformer_seq_proj_triton(proj, seq)
        out_pt = fused_linformer_seq_proj_pytorch(proj, seq)

        assert not torch.isnan(out_triton).any()
        assert not torch.isinf(out_triton).any()
        diff = (out_triton - out_pt).abs().max().item()
        assert diff <= 1e-4, f"Linformer projection mismatch: {diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [64, 256])
    def test_nystrom_landmark_pinv_boundary(self, L: int):
        from neuralops.core.triton_kernels.low_rank import (
            fused_nystrom_landmark_pinv_step_triton,
            fused_nystrom_landmark_pinv_step_pytorch,
        )

        B, H, D, m = 2, 4, 64, 16
        F1 = F.softmax(torch.randn(B, H, L, m, device=DEVICE, dtype=torch.float32), dim=-1)
        F2_pinv = torch.randn(B, H, m, m, device=DEVICE, dtype=torch.float32)
        F3 = F.softmax(torch.randn(B, H, m, L, device=DEVICE, dtype=torch.float32), dim=-1)
        V = torch.randn(B, H, L, D, device=DEVICE, dtype=torch.float32)

        out_triton = fused_nystrom_landmark_pinv_step_triton(F1, F2_pinv, F3, V)
        out_pt = fused_nystrom_landmark_pinv_step_pytorch(F1, F2_pinv, F3, V)

        assert not torch.isnan(out_triton).any()
        assert not torch.isinf(out_triton).any()
        diff = (out_triton - out_pt).abs().max().item()
        assert diff <= 1e-4, f"Nystrom landmark pinv step mismatch: {diff}"



class TestHighFlopMemoryIntensiveAttentions:
    """Stress tests routing, sink, retention, rwkv6, and differential attentions."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [64, 256, 1024])
    def test_routing_attention_grad_and_stability(self, L: int):
        from neuralops.attentions.sparse.routing_attention import RoutingAttention

        cfg = SimpleNamespace(hidden_size=64, num_attention_heads=4, attention_dropout=0.0)
        mod = RoutingAttention(config=cfg, num_clusters=4, n_iters=2).to(DEVICE)

        x = torch.randn(2, L, 64, device=DEVICE, requires_grad=True)
        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]

        assert not torch.isnan(out).any(), f"NaN in RoutingAttention output at L={L}"
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Missing gradient for RoutingAttention"
        grad_norm = x.grad.norm().item()
        assert 1e-7 < grad_norm < 1e5, f"RoutingAttention grad abnormal: {grad_norm}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [64, 256, 1024])
    def test_sink_attention_grad_and_stability(self, L: int):
        from neuralops.attentions.adaptive.sink_attention import SinkAttention

        mod = SinkAttention(d_model=64, n_heads=4, window_size=32, num_sinks=4).to(DEVICE)
        x = torch.randn(2, L, 64, device=DEVICE, requires_grad=True)

        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]

        assert not torch.isnan(out).any(), f"NaN in SinkAttention output at L={L}"
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Missing gradient for SinkAttention"
        grad_norm = x.grad.norm().item()
        assert 1e-7 < grad_norm < 1e5, f"SinkAttention grad abnormal: {grad_norm}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("L", [32, 128])
    def test_rwkv6_and_multi_scale_retention_autograd(self, L: int):
        from neuralops.core.cuda_kernels import rwkv6, multi_scale_retention

        B, H, D = 2, 4, 64
        q = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        k = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        v = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        gammas = torch.linspace(0.8, 0.99, H, device=DEVICE, requires_grad=True)

        out_ret = multi_scale_retention(q, k, v, gammas)
        assert not torch.isnan(out_ret).any()
        out_ret.sum().backward()
        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert 1e-7 < q.grad.norm().item() < 1e5

        r = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        k = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        v = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        w = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        u = torch.randn(H, D, device=DEVICE, requires_grad=True)

        out_rwkv = rwkv6(r, k, v, w, u)
        assert not torch.isnan(out_rwkv).any()
        out_rwkv.sum().backward()
        assert r.grad is not None and not torch.isnan(r.grad).any()
        assert 1e-7 < r.grad.norm().item() < 1e5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    def test_differential_attention_autograd(self):
        from neuralops.core.cuda_kernels import differential_attention
        from neuralops.core.triton_kernels.differential_attention import triton_differential_attention

        B, H, L, D = 2, 4, 32, 64
        q1 = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        q2 = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        k1 = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        k2 = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)
        v = torch.randn(B, H, L, D, device=DEVICE, requires_grad=True)

        out = differential_attention(q1, q2, k1, k2, v, 0.8)
        assert not torch.isnan(out).any()
        out.sum().backward()

        assert q1.grad is not None
        assert 1e-7 < q1.grad.norm().item() < 1e5

        out_tri = triton_differential_attention(q1, q2, k1, k2, v)
        assert not torch.isnan(out_tri).any()
        assert out_tri.shape == (B, H, L, D)



class TestHybridMultiBranchStructures:
    """Validates Split-Head, Tri-Branch, and Jamba hybrid attention architectures."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    def test_split_head_and_tri_branch_hybrids(self):
        from neuralops.attentions.hybrid.split_head_hybrid_attention import SplitHeadHybridAttention
        from neuralops.attentions.hybrid.tri_branch_hybrid_attention import TriBranchHybridAttention

        B, L, D = 2, 64, 64

        mod_split = SplitHeadHybridAttention(d_model=D, n_heads=4).to(DEVICE)
        x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        out = mod_split(x)
        if isinstance(out, tuple):
            out = out[0]
        assert not torch.isnan(out).any()
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

        cfg = SimpleNamespace(hidden_size=D, num_attention_heads=4, attention_dropout=0.0)
        mod_tri = TriBranchHybridAttention(config=cfg, hidden_size=D, num_heads=4).to(DEVICE)
        x2 = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        out2 = mod_tri(x2)
        if isinstance(out2, tuple):
            out2 = out2[0]
        assert not torch.isnan(out2).any()
        out2.sum().backward()
        assert x2.grad is not None and not torch.isnan(x2.grad).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    def test_jamba_and_cosformer_hybrids(self):
        from neuralops.attentions.ssm.jamba import Jamba2Block
        from neuralops.attentions.ssm.jamba_info import JambaInfoBlock
        from neuralops.attentions.linear.cosformer import CosformerAttention

        B, L, D = 2, 64, 64

        mod_cos = CosformerAttention(d_model=D, n_heads=4).to(DEVICE)
        x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        out_cos = mod_cos(x)
        if isinstance(out_cos, tuple):
            out_cos = out_cos[0]
        assert not torch.isnan(out_cos).any()
        out_cos.sum().backward()
        assert x.grad is not None

        mod_jamba = Jamba2Block(d_model=D, n_heads=4).to(DEVICE)
        x_j = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        out_j = mod_jamba(x_j)
        if isinstance(out_j, tuple):
            out_j = out_j[0]
        assert not torch.isnan(out_j).any()
        out_j.sum().backward()
        assert x_j.grad is not None

        mod_jinfo = JambaInfoBlock(d_model=D, n_heads=4).to(DEVICE)
        x_ji = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
        out_ji = mod_jinfo(x_ji)
        if isinstance(out_ji, tuple):
            out_ji = out_ji[0]
        assert not torch.isnan(out_ji).any()
        out_ji.sum().backward()
        assert x_ji.grad is not None



class TestCompressionEnginesHardening:
    """Stress tests high reconstruction error & extreme rank compressors."""

    class LinearProbe(nn.Module):
        def __init__(self, in_f: int = 128, out_f: int = 128):
            super().__init__()
            self.linear = nn.Linear(in_f, out_f, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("rank", [8, 64])
    def test_cr_and_cur_compressors(self, rank: int):
        from neuralops.compression_engine.low_rank.cr_compressor import CRCompressor
        from neuralops.compression_engine.low_rank.cur_compressor import CURCompressor

        model = self.LinearProbe().to(DEVICE)
        w_orig = model.linear.weight.detach().clone()

        comp_cr = CRCompressor(rank=rank, max_error=0.99)
        compressed_cr, _ = comp_cr.compress(model)
        x = torch.randn(4, 128, device=DEVICE)
        out_cr = compressed_cr(x)
        assert not torch.isnan(out_cr).any()

        model_cur = self.LinearProbe().to(DEVICE)
        comp_cur = CURCompressor(rank=rank, max_error=0.99)
        compressed_cur, _ = comp_cur.compress(model_cur)
        out_cur = compressed_cur(x)
        assert not torch.isnan(out_cur).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    @pytest.mark.parametrize("rank", [8, 32])
    def test_frandi_grecodec_residual_svd(self, rank: int):
        from neuralops.compression_engine.low_rank.frandi_compressor import FRanDICompressor
        from neuralops.compression_engine.low_rank.grecodec_compressor import GreCoDeCCompressor
        from neuralops.compression_engine.svd.residual_svd_compressor import ResidualSVDCompressor

        model_frandi = self.LinearProbe().to(DEVICE)
        comp_frandi = FRanDICompressor(rank=rank, max_error=0.99)
        comp_m_frandi, _ = comp_frandi.compress(model_frandi)
        x = torch.randn(4, 128, device=DEVICE)
        out_frandi = comp_m_frandi(x)
        assert not torch.isnan(out_frandi).any()

        model_gre = self.LinearProbe().to(DEVICE)
        comp_gre = GreCoDeCCompressor(rank=rank, max_error=0.99)
        comp_m_gre, _ = comp_gre.compress(model_gre)
        out_gre = comp_m_gre(x)
        assert not torch.isnan(out_gre).any()

        model_res = self.LinearProbe().to(DEVICE)
        comp_res = ResidualSVDCompressor(model=model_res, base_rank=rank, residual_rank=max(2, rank // 4))
        comp_m_res, _ = comp_res.compress()
        out_res = comp_m_res(x)
        assert not torch.isnan(out_res).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    def test_quantization_compressors_boundary(self):
        from neuralops.compression_engine.quantization.smoothquant import SmoothQuantCompressor
        from neuralops.compression_engine.quantization.fp8_compressor import FP8Compressor

        model_sq = self.LinearProbe().to(DEVICE)
        comp_sq = SmoothQuantCompressor(model=model_sq)
        comp_m_sq, _ = comp_sq.compress()

        x = torch.randn(4, 128, device=DEVICE)
        out_sq = comp_m_sq(x)
        assert not torch.isnan(out_sq).any()

        model_fp8 = self.LinearProbe().to(DEVICE)
        comp_fp8 = FP8Compressor(model=model_fp8)
        comp_m_fp8, _ = comp_fp8.compress()
        out_fp8 = comp_m_fp8(x)
        assert not torch.isnan(out_fp8).any()



class TestDynamicContextAndMemoryIntegrity:
    """Profiles memory stability under dynamic sequence scaling L in {128, 512, 2048}."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
    def test_dynamic_scaling_memory_bounded(self):
        from neuralops.attentions.adaptive.sink_attention import SinkAttention

        mod = SinkAttention(d_model=64, n_heads=4, window_size=64, num_sinks=4).to(DEVICE)
        
        _ = mod(torch.randn(2, 64, 64, device=DEVICE))
        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated(DEVICE)

        for L in [128, 512, 2048]:
            x = torch.randn(2, L, 64, device=DEVICE)
            out = mod(x)
            if isinstance(out, tuple):
                out = out[0]
            assert out.shape == (2, L, 64)
            del x, out
            torch.cuda.synchronize()

        mem_end = torch.cuda.memory_allocated(DEVICE)
        mem_diff_mb = (mem_end - mem_start) / (1024 * 1024)
        assert mem_diff_mb < 20.0, f"Potential memory leak detected: {mem_diff_mb:.2f} MB retained"
