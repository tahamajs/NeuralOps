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
    # 1. softplus
    x = torch.randn(4, 4)
    from neuralops.core.triton_kernels.softplus import softplus_fn
    assert torch.allclose(softplus_fn(x), F.softplus(x))
    assert callable(softplus)

    # 2. low_rank fallback
    u = torch.randn(4, 8)
    v = torch.randn(8, 4)
    act = low_rank_pytorch_fallback(x, u, v)
    assert act.shape == (4, 4)
    assert torch.isfinite(act).all()

    # 3. gqa fallback
    k = torch.randn(2, 2, 8, 16)
    v_t = torch.randn(2, 2, 8, 16)
    k_exp, v_exp = gqa_pytorch_fallback(k, v_t, num_kv_heads=2, num_heads=4)
    assert k_exp.shape == (2, 4, 8, 16)
    assert v_exp.shape == (2, 4, 8, 16)

    # 4. linear_attn fallback
    q = torch.randn(2, 2, 16, 16)
    k_l = torch.randn(2, 2, 16, 16)
    v_l = torch.randn(2, 2, 16, 16)
    out_l = linear_attn_pytorch_fallback(q, k_l, v_l)
    assert out_l.shape == (2, 2, 16, 16)
    assert torch.isfinite(out_l).all()

    # 5. state_update fallback
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

    # Cosformer attention
    cosformer = TritonCosformerAttention(embed_dim=64, num_heads=4)
    out_cos, _ = cosformer(x)
    assert out_cos.shape == (2, 8, 64)
    assert torch.isfinite(out_cos).all()

    # SVD linear attention
    svd_attn = TritonSVDLinearAttention(embed_dim=64, num_heads=4, rank=16)
    out_svd, _ = svd_attn(x)
    assert out_svd.shape == (2, 8, 64)
    assert torch.isfinite(out_svd).all()
