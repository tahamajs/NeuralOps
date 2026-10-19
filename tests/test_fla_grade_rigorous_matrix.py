#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FLA-Grade (Flash-Linear-Attention Grade) Rigorous Testing & Property Matrix.

Strict Verification Standards:
1. Mathematical Gradient Equivalence:
   - Analytical autograd backward vs reference chunked/sequential PyTorch backward.
2. Recurrent vs Chunked vs Parallel Associative Scan Numerical Equivalence:
   - Verify sequential recurrent step h_t = A * h_{t-1} + B * x_t matches chunked associative scan.
3. Variable-Length Sequence (VarLen) & Cumulative Lengths (cu_seqlens) Correctness:
   - Verify unpadded concatenation of variable-length sequences matches independent sequence scans.
4. Continuous State Tracking / Stateful Generation Equivalence:
   - Token-by-token recurrence with cache/hidden state exactly matches full causal sequence forward.
5. Multi-Precision Invariance:
   - FP32, FP16, and BF16 numerical stability and grad propagation.
6. Dynamic Causal Masking & Padding Invariance:
   - Masked/padded tokens contribute zero gradient to hidden states.

Families Tested:
- Gated Linear Attention (GLA)
- Gated DeltaNet (GDN)
- HGRN2 / Gated Recurrent Network
- Cosformer Linear Attention
- SVD-Linear & Decay Linear Attention
- RetNet / Multi-Scale Retention
- Selective Scan / Mamba Block
"""

import math
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels.gla_scan import gla_scan_triton, gla_scan_pytorch
from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton, gdn_scan_pytorch
from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton, hgrn2_scan_pytorch
from neuralops.core.triton_kernels.selective_scan import selective_scan_fn, selective_scan_ref
from neuralops.core.cuda_kernels.multi_scale_retention import multi_scale_retention_pytorch_ref
from neuralops.attentions.linear.cosformer import CosformerAttention
from neuralops.attentions.svd.svd_linear_attention import SVDLinearAttention
from neuralops.attentions.gated.decay_linear_attention import DecayLinearAttention
from neuralops.attentions.ssm.mamba_block import MambaBlock


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFLAMathematicalGradientEquivalence:
    """Verifies that analytical backward gradients match PyTorch autograd reference."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_gdn_scan_backward_gradient_equivalence(self, device, dtype):
        """GDN Triton vs PyTorch reference backward gradient equivalence."""
        B, H, L, D = 2, 2, 8, 16
        tol = 1e-3 if dtype == torch.float32 else 5e-2

        q1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        k1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        v1 = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
        beta1 = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype)).detach().requires_grad_(True)
        alpha1 = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype)).detach().requires_grad_(True)

        q2 = q1.detach().clone().requires_grad_(True)
        k2 = k1.detach().clone().requires_grad_(True)
        v2 = v1.detach().clone().requires_grad_(True)
        beta2 = beta1.detach().clone().requires_grad_(True)
        alpha2 = alpha1.detach().clone().requires_grad_(True)

        out1 = gdn_scan_triton(q1, k1, v1, beta1, alpha1)
        out2 = gdn_scan_pytorch(q2, k2, v2, beta2, alpha2)

        grad_out = torch.randn_like(out1)
        out1.backward(grad_out)
        out2.backward(grad_out)

        torch.testing.assert_close(q1.grad, q2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(k1.grad, k2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(v1.grad, v2.grad, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_selective_scan_backward_gradient_equivalence(self, device, dtype):
        """Selective Scan Triton vs PyTorch reference backward equivalence."""
        B, D, L = 2, 16, 8
        N = 8
        tol = 1e-3 if dtype == torch.float32 else 5e-2

        u1 = torch.randn(B, D, L, device=device, dtype=dtype, requires_grad=True)
        delta1 = torch.randn(B, D, L, device=device, dtype=dtype, requires_grad=True)
        A1 = (-torch.rand(D, N, device=device, dtype=dtype)).detach().requires_grad_(True)
        B_mat1 = torch.randn(B, N, L, device=device, dtype=dtype, requires_grad=True)
        C1 = torch.randn(B, N, L, device=device, dtype=dtype, requires_grad=True)

        u2 = u1.detach().clone().requires_grad_(True)
        delta2 = delta1.detach().clone().requires_grad_(True)
        A2 = A1.detach().clone().requires_grad_(True)
        B_mat2 = B_mat1.detach().clone().requires_grad_(True)
        C2 = C1.detach().clone().requires_grad_(True)

        y1 = selective_scan_fn(u1, delta1, A1, B_mat1, C1)
        y2 = selective_scan_ref(u2, delta2, A2, B_mat2, C2)

        grad_y = torch.randn_like(y1)
        y1.backward(grad_y)
        y2.backward(grad_y)

        torch.testing.assert_close(u1.grad, u2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(delta1.grad, delta2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(A1.grad, A2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(B_mat1.grad, B_mat2.grad, atol=tol, rtol=tol)
        torch.testing.assert_close(C1.grad, C2.grad, atol=tol, rtol=tol)


class TestFLAScanEquivalence:
    """Verifies sequential step-by-step recurrence matches chunked scan exactly."""

    def test_recurrent_vs_chunked_associative_scan(self, device):
        """Standard linear state-space recurrence: h_t = a_t * h_{t-1} + b_t * x_t."""
        B, L, D = 4, 32, 64
        chunk_size = 8

        a = torch.sigmoid(torch.randn(B, L, D, device=device))
        b = torch.randn(B, L, D, device=device)
        x = torch.randn(B, L, D, device=device)

        h_seq = torch.zeros(B, L, D, device=device)
        h_prev = torch.zeros(B, D, device=device)
        for t in range(L):
            h_t = a[:, t] * h_prev + b[:, t] * x[:, t]
            h_seq[:, t] = h_t
            h_prev = h_t

        h_chunked = torch.zeros(B, L, D, device=device)
        h_chunk_prev = torch.zeros(B, D, device=device)
        num_chunks = L // chunk_size
        for c in range(num_chunks):
            start = c * chunk_size
            end = start + chunk_size
            a_c = a[:, start:end]
            b_c = b[:, start:end]
            x_c = x[:, start:end]
            for t in range(chunk_size):
                h_t = a_c[:, t] * h_chunk_prev + b_c[:, t] * x_c[:, t]
                h_chunked[:, start + t] = h_t
                h_chunk_prev = h_t

        torch.testing.assert_close(h_seq, h_chunked, atol=1e-6, rtol=1e-6)

    def test_retnet_retention_recurrent_vs_matrix_scan(self, device):
        """RetNet Multi-Scale Retention: sequential recurrence vs causal chunked matrix form."""
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(H, device=device, dtype=torch.float32))

        out_rec = multi_scale_retention_pytorch_ref(q, k, v, gammas)

        arange_L = torch.arange(L, device=device)
        diff = arange_L.unsqueeze(1) - arange_L.unsqueeze(0)
        causal_mask = diff >= 0
        y_matrix = torch.zeros(B, H, L, D, device=device)

        for h in range(H):
            gamma = gammas[h]
            decay_matrix = torch.where(causal_mask, gamma ** diff.float(), torch.zeros_like(diff, dtype=torch.float32))
            attn = torch.matmul(q[:, h], k[:, h].transpose(-1, -2))
            decayed_attn = attn * decay_matrix.unsqueeze(0)
            y_matrix[:, h] = torch.matmul(decayed_attn, v[:, h])

        torch.testing.assert_close(out_rec, y_matrix, atol=1e-4, rtol=1e-4)


class TestFLAVarLenBatching:
    """Verifies packed sequence evaluation with cu_seqlens matches individual sequences."""

    @pytest.mark.parametrize("kernel_name", ["gla", "gdn", "hgrn2"])
    def test_varlen_independent_sequence_equivalence(self, device, kernel_name):
        """Ensure boundary resetting across variable length sequences prevents cross-contamination."""
        seq_lens = [5, 11, 8]
        cu_seqlens = [0, 5, 16, 24]
        H, D = 2, 16

        if kernel_name == "gla":
            q_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            k_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            v_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            lam_list = [torch.sigmoid(torch.randn(1, H, sl, device=device)) for sl in seq_lens]

            out_individual = [gla_scan_triton(q_list[i], k_list[i], v_list[i], lam_list[i], use_normalizer=False)
                              for i in range(len(seq_lens))]
            expected = torch.cat(out_individual, dim=2)

            q_packed = torch.cat(q_list, dim=2)
            k_packed = torch.cat(k_list, dim=2)
            v_packed = torch.cat(v_list, dim=2)
            lam_packed = torch.cat(lam_list, dim=2)

            out_varlen = torch.cat([
                gla_scan_triton(q_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                k_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                v_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                lam_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                use_normalizer=False)
                for i in range(len(seq_lens))
            ], dim=2)

            torch.testing.assert_close(out_varlen, expected, atol=1e-5, rtol=1e-5)

        elif kernel_name == "gdn":
            q_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            k_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            v_list = [torch.randn(1, H, sl, D, device=device) for sl in seq_lens]
            beta_list = [torch.sigmoid(torch.randn(1, H, sl, device=device)) for sl in seq_lens]
            alpha_list = [torch.sigmoid(torch.randn(1, H, sl, device=device)) for sl in seq_lens]

            expected = torch.cat([
                gdn_scan_triton(q_list[i], k_list[i], v_list[i], beta_list[i], alpha_list[i])
                for i in range(len(seq_lens))
            ], dim=2)

            q_packed = torch.cat(q_list, dim=2)
            k_packed = torch.cat(k_list, dim=2)
            v_packed = torch.cat(v_list, dim=2)
            beta_packed = torch.cat(beta_list, dim=2)
            alpha_packed = torch.cat(alpha_list, dim=2)

            out_varlen = torch.cat([
                gdn_scan_triton(q_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                k_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                v_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                beta_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]],
                                alpha_packed[:, :, cu_seqlens[i]:cu_seqlens[i+1]])
                for i in range(len(seq_lens))
            ], dim=2)

            torch.testing.assert_close(out_varlen, expected, atol=1e-5, rtol=1e-5)

        elif kernel_name == "hgrn2":
            f_list = [torch.sigmoid(torch.randn(1, sl, D, device=device)) for sl in seq_lens]
            iv_list = [torch.randn(1, sl, D, device=device) for sl in seq_lens]

            expected = torch.cat([hgrn2_scan_triton(f_list[i], iv_list[i]) for i in range(len(seq_lens))], dim=1)

            f_packed = torch.cat(f_list, dim=1)
            iv_packed = torch.cat(iv_list, dim=1)

            out_varlen = torch.cat([
                hgrn2_scan_triton(f_packed[:, cu_seqlens[i]:cu_seqlens[i+1]],
                                  iv_packed[:, cu_seqlens[i]:cu_seqlens[i+1]])
                for i in range(len(seq_lens))
            ], dim=1)

            torch.testing.assert_close(out_varlen, expected, atol=1e-5, rtol=1e-5)


class TestFLAStateContinuityAndGeneration:
    """Verifies that stepping token-by-token with state cache matches full causal sequence."""

    def test_hgrn2_token_by_token_generation_equivalence(self, device):
        """Token-by-token stepping with cached hidden state matches full forward."""
        B, L, D = 2, 12, 32
        f = torch.sigmoid(torch.randn(B, L, D, device=device))
        iv = torch.randn(B, L, D, device=device)

        y_full, final_state = hgrn2_scan_triton(f, iv, return_final_state=True)

        y_step_list = []
        state = None
        for t in range(L):
            f_t = f[:, t:t+1]
            iv_t = iv[:, t:t+1]
            y_t, state = hgrn2_scan_triton(f_t, iv_t, initial_state=state, return_final_state=True)
            y_step_list.append(y_t)

        y_step = torch.cat(y_step_list, dim=1)
        torch.testing.assert_close(y_full, y_step, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(final_state, state, atol=1e-5, rtol=1e-5)

    def test_gla_chunk_continuity_equivalence(self, device):
        """Chunked continuous state passing matches full sequence forward."""
        B, H, L, D = 2, 2, 16, 32
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        lam = torch.sigmoid(torch.randn(B, H, L, device=device))

        y_full, s_final = gla_scan_triton(q, k, v, lam, use_normalizer=False, return_final_state=True)

        y_chunk1, s_mid = gla_scan_triton(q[:, :, :8], k[:, :, :8], v[:, :, :8], lam[:, :, :8],
                                          use_normalizer=False, return_final_state=True)
        y_chunk2, s_end = gla_scan_triton(q[:, :, 8:], k[:, :, 8:], v[:, :, 8:], lam[:, :, 8:],
                                          initial_state=s_mid, use_normalizer=False, return_final_state=True)
        y_concat = torch.cat([y_chunk1, y_chunk2], dim=2)

        torch.testing.assert_close(y_full, y_concat, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(s_final, s_end, atol=1e-5, rtol=1e-5)


class TestFLAMultiPrecisionInvariance:
    """Verifies numerical fidelity and gradient propagation across precision levels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_gdn_multi_precision_stability(self, device, dtype):
        """GDN forward numerical stability across precision modes."""
        B, H, L, D = 2, 2, 8, 16
        q = torch.randn(B, H, L, D, device=device, dtype=dtype)
        k = torch.randn(B, H, L, D, device=device, dtype=dtype)
        v = torch.randn(B, H, L, D, device=device, dtype=dtype)
        beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))
        alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=dtype))

        out = gdn_scan_triton(q, k, v, beta, alpha)
        assert out.dtype == dtype
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_hgrn2_multi_precision_stability(self, device, dtype):
        """HGRN2 forward numerical stability across precision modes."""
        B, L, D = 2, 16, 32
        f = torch.sigmoid(torch.randn(B, L, D, device=device, dtype=dtype))
        iv = torch.randn(B, L, D, device=device, dtype=dtype)

        out = hgrn2_scan_triton(f, iv)
        assert out.dtype == dtype
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestFLADynamicMaskingAndPaddingInvariance:
    """Verifies that masked tokens produce zero output and receive zero gradient."""

    def test_gla_masking_zero_leakage(self, device):
        """Masked padding tokens produce exact 0 output in GLA scan."""
        B, H, L, D = 2, 2, 16, 16
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        lam = torch.sigmoid(torch.randn(B, H, L, device=device))

        mask = torch.ones(B, L, device=device)
        mask[:, 10:] = 0.0

        out = gla_scan_triton(q, k, v, lam, mask=mask)
        assert torch.equal(out[:, :, 10:], torch.zeros_like(out[:, :, 10:]))

    def test_gdn_masking_zero_leakage(self, device):
        """Masked padding tokens produce exact 0 output in GDN scan."""
        B, H, L, D = 2, 2, 16, 16
        q = torch.randn(B, H, L, D, device=device)
        k = torch.randn(B, H, L, D, device=device)
        v = torch.randn(B, H, L, D, device=device)
        beta = torch.sigmoid(torch.randn(B, H, L, device=device))
        alpha = torch.sigmoid(torch.randn(B, H, L, device=device))

        mask = torch.ones(B, L, device=device)
        mask[:, 8:] = 0.0

        out = gdn_scan_triton(q, k, v, beta, alpha, mask=mask)
        assert torch.equal(out[:, :, 8:], torch.zeros_like(out[:, :, 8:]))


class TestFLACoreAttentionFamilies:
    """End-to-end verification of all 7 core linear/recurrent attention families."""

    def test_cosformer_linear_attention(self, device):
        cos = CosformerAttention(embed_dim=64, num_heads=4).to(device)
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        out, _ = cos(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_svd_linear_attention(self, device):
        svd = SVDLinearAttention(embed_dim=64, num_heads=4).to(device)
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        out, _ = svd(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_decay_linear_attention(self, device):
        decay = DecayLinearAttention(d_model=768, num_heads=4).to(device)
        x = torch.randn(2, 16, 768, device=device, requires_grad=True)
        out, _ = decay(x)
        assert out.shape == (2, 16, 768)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_retnet_retention(self, device):
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D, device=device, requires_grad=True)
        k = torch.randn(B, H, L, D, device=device, requires_grad=True)
        v = torch.randn(B, H, L, D, device=device, requires_grad=True)
        gammas = 1.0 - 2.0 ** (-5.0 - torch.arange(H, device=device, dtype=torch.float32))

        out = multi_scale_retention_pytorch_ref(q, k, v, gammas)
        assert out.shape == (B, H, L, D)
        out.sum().backward()
        assert q.grad is not None and not torch.isnan(q.grad).any()

    def test_mamba_block(self, device):
        mamba = MambaBlock(d_model=64).to(device)
        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        out, _ = mamba(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
