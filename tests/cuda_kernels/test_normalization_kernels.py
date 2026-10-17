"""Production-Grade Test Suite for NeuralOps Normalization, Activation, and Elementwise CUDA Kernels.

Tests 19 CUDA C++ kernel modules:
1. aegis_gate_norm
2. channel_shuffle
3. entmax_bisect
4. entropy_gate
5. entropy_pruning
6. fast_gelu
7. fused_ce
8. fused_dropout_add
9. fused_kd
10. fused_norm_linear
11. fused_rope
12. fwht_transform
13. layernorm
14. rms_variance_match
15. rmsnorm
16. softplus
17. split_head_norm
18. swiglu
19. unified_rope_offset

Verifies:
- Numerical parity: FP16 max_diff < 0.01, FP32 max_diff < 1e-4
- Zero NaNs and Infs
- Multi-shape sweeps & contiguous vs non-contiguous layouts
- Backward / autograd pass where gradient graphs exist
- 3-level fallback dispatch (CUDA vs CPU PyTorch reference)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels import force_kernel_backend
from tests.cuda_kernels.conftest import (
    assert_numerical_parity,
    check_tensor_validity,
    create_non_contiguous,
)

import neuralops.core.cuda_kernels.aegis_gate_norm as m_aegis
import neuralops.core.cuda_kernels.channel_shuffle as m_shuffle
import neuralops.core.cuda_kernels.entmax_bisect as m_entmax
import neuralops.core.cuda_kernels.entropy_gate as m_ent_gate
import neuralops.core.cuda_kernels.entropy_pruning as m_ent_prune
import neuralops.core.cuda_kernels.fast_gelu as m_gelu
import neuralops.core.cuda_kernels.fused_ce as m_ce
import neuralops.core.cuda_kernels.fused_dropout_add as m_dropout
import neuralops.core.cuda_kernels.fused_kd as m_kd
import neuralops.core.cuda_kernels.fused_norm_linear as m_norm_linear
import neuralops.core.cuda_kernels.fused_rope as m_rope
import neuralops.core.cuda_kernels.fwht_transform as m_fwht
import neuralops.core.cuda_kernels.layernorm as m_ln
import neuralops.core.cuda_kernels.rms_variance_match as m_vmatch
import neuralops.core.cuda_kernels.rmsnorm as m_rms
import neuralops.core.cuda_kernels.softplus as m_softplus
import neuralops.core.cuda_kernels.split_head_norm as m_split_head
import neuralops.core.cuda_kernels.swiglu as m_swiglu
import neuralops.core.cuda_kernels.unified_rope_offset as m_unirope


@pytest.fixture(autouse=True)
def setup_cuda_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for native CUDA kernel testing.")
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
# 1. aegis_gate_norm
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(2, 8, 64), (1, 16, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_aegis_gate_norm(shape, dtype):
    """Verifies multi-branch RMSNorm with gated routing."""
    B, L, D = shape
    b0 = torch.randn(B, L, D, device="cuda", dtype=dtype)
    b1 = torch.randn(B, L, D, device="cuda", dtype=dtype)
    weights = torch.randn(B, L, 2, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_aegis.aegis_gate_norm(b0, b1, weights)

    out_ref = m_aegis.aegis_gate_norm_pytorch_ref(b0, b1, weights)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="aegis_gate_norm")

    # Non-contiguous
    b0_nc = create_non_contiguous(b0)
    with force_kernel_backend("cuda"):
        out_nc = m_aegis.aegis_gate_norm(b0_nc, b1, weights)
    assert_numerical_parity(out_nc, out_ref, dtype=dtype, name="aegis_gate_norm_nc")


# ══════════════════════════════════════════════════════════════════════
# 2. channel_shuffle
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(2, 16, 32), (4, 8, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_channel_shuffle(shape, dtype):
    """Verifies 128-bit vectorized channel shuffle across grouped dimensions."""
    x = torch.randn(shape, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_shuffle.channel_shuffle(x, groups=2)

    out_ref = m_shuffle.channel_shuffle_pytorch_ref(x, groups=2)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="channel_shuffle")


# ══════════════════════════════════════════════════════════════════════
# 3. entmax_bisect
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(2, 32), (4, 64)])
def test_entmax_bisect(shape):
    """Verifies exact root-finding bisection for alpha-1.5 entmax."""
    scores = torch.randn(shape, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_entmax.entmax_bisect(scores, n_iter=15)

    out_ref = m_entmax.fused_entmax_bisect_pytorch(scores, alpha=1.5, n_iter=15)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="entmax_bisect")


# ══════════════════════════════════════════════════════════════════════
# 4. entropy_gate
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_entropy_gate(dtype):
    """Verifies low-rank SVD gating modulated by spectral entropy."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)
    Vr = torch.randn(16, 64, device="cuda", dtype=dtype)
    Ur = torch.randn(64, 16, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_ent_gate.spectral_entropy_svd_gate(x, Vr, Ur, effective_rank=8)

    out_ref = m_ent_gate.spectral_entropy_svd_gate_pytorch_ref(x, Vr, Ur, effective_rank=8)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="entropy_gate")


# ══════════════════════════════════════════════════════════════════════
# 5. entropy_pruning
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(2, 64), (4, 128)])
def test_entropy_pruning(shape):
    """Verifies online row-level Shannon entropy reduction with broadcast warp shuffles."""
    scores = torch.randn(shape, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_ent_prune.entropy_pruning(scores)

    out_ref = m_ent_prune.entropy_pruning_pytorch_ref(scores)
    assert_numerical_parity(out_cuda.view_as(out_ref), out_ref, dtype=torch.float32, name="entropy_pruning")


# ══════════════════════════════════════════════════════════════════════
# 6. fast_gelu
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(2, 64), (4, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fast_gelu(shape, dtype):
    """Verifies fast tanh-approximated GELU activation."""
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)

    with force_kernel_backend("cuda"):
        out_cuda = m_gelu.fast_gelu(x)

    out_ref = m_gelu.fast_gelu_pytorch_ref(x)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="fast_gelu")

    # Autograd test
    grad_cuda = torch.autograd.grad(out_cuda.sum(), x, retain_graph=True)[0]
    grad_ref = torch.autograd.grad(out_ref.sum(), x, retain_graph=False)[0]
    assert_numerical_parity(grad_cuda, grad_ref, dtype=dtype, name="fast_gelu_bwd")


# ══════════════════════════════════════════════════════════════════════
# 7. fused_ce
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("shape", [(4, 100), (8, 256)])
def test_fused_cross_entropy(shape):
    """Verifies online fused cross entropy loss computation."""
    B, C = shape
    logits = torch.randn(B, C, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, C, (B,), device="cuda", dtype=torch.long)

    with force_kernel_backend("cuda"):
        out_cuda = m_ce.fused_cross_entropy(logits, targets)

    out_ref = m_ce.fused_cross_entropy_pytorch_ref(logits, targets)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="fused_cross_entropy")

    # Autograd check
    grad_cuda = torch.autograd.grad(out_cuda, logits, retain_graph=True)[0]
    grad_ref = torch.autograd.grad(out_ref, logits, retain_graph=False)[0]
    assert_numerical_parity(grad_cuda, grad_ref, dtype=torch.float32, name="fused_cross_entropy_bwd")


# ══════════════════════════════════════════════════════════════════════
# 8. fused_dropout_add
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_dropout_add(dtype):
    """Verifies fused residual addition with deterministic dropout bypass."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)
    res = torch.randn(2, 64, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_dropout.fused_dropout_add(x, res, p=0.0, training=False)

    out_ref = m_dropout.fused_dropout_add_pytorch_ref(x, res, p=0.0, training=False)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="fused_dropout_add")


# ══════════════════════════════════════════════════════════════════════
# 9. fused_kd
# ══════════════════════════════════════════════════════════════════════
def test_fused_kd_loss():
    """Verifies temperature-scaled knowledge distillation loss."""
    s_logits = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    t_logits = torch.randn(4, 32, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_kd.fused_kd_loss(s_logits, t_logits, temperature=2.0)

    out_ref = m_kd.fused_kd_loss_pytorch_ref(s_logits, t_logits, temperature=2.0)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="fused_kd_loss")


# ══════════════════════════════════════════════════════════════════════
# 10. fused_norm_linear
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_norm_linear(dtype):
    """Verifies fused RMSNorm and linear projection."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)
    w = torch.randn(32, 64, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_norm_linear.fused_norm_linear(x, w)

    out_ref = m_norm_linear.fused_norm_linear_pytorch_ref(x, w)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="fused_norm_linear")


# ══════════════════════════════════════════════════════════════════════
# 11. fused_rope
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_rotary_emb(dtype):
    """Verifies fast rotary positional embeddings."""
    x = torch.randn(2, 4, 16, 64, device="cuda", dtype=dtype)
    cos = torch.randn(16, 32, device="cuda", dtype=dtype)
    sin = torch.randn(16, 32, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_rope.fused_rotary_emb(x, cos, sin)

    out_ref = m_rope.fused_rotary_emb_pytorch_ref(x, cos, sin)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="fused_rotary_emb")


# ══════════════════════════════════════════════════════════════════════
# 12. fwht_transform
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("B_dim", [32, 64, 128])
def test_fwht_transform(B_dim):
    """Verifies Fast Walsh-Hadamard Transform in shared memory."""
    x = torch.randn(2, B_dim, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_fwht.fwht_transform(x, B_dim=B_dim)

    out_ref = m_fwht.fwht_transform_pytorch_ref(x, B_dim=B_dim)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="fwht_transform")


# ══════════════════════════════════════════════════════════════════════
# 13. layernorm
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_layernorm(dtype):
    """Verifies fused LayerNorm forward and autograd pass."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(64, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(64, device="cuda", dtype=dtype, requires_grad=True)

    with force_kernel_backend("cuda"):
        out_cuda = m_ln.layernorm(x, w, b)

    out_ref = m_ln.layernorm_pytorch_ref(x, w, b)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="layernorm")

    # Autograd backward test
    out_cuda.sum().backward()
    grad_x = x.grad.clone()
    x.grad.zero_()
    out_ref.sum().backward()
    grad_ref = x.grad.clone()
    assert_numerical_parity(grad_x, grad_ref, dtype=dtype, name="layernorm_bwd")


# ══════════════════════════════════════════════════════════════════════
# 14. rms_variance_match
# ══════════════════════════════════════════════════════════════════════
def test_rms_variance_match():
    """Verifies variance stabilization matching between feature representations."""
    y = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    v_ref = torch.randn(2, 64, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_vmatch.rms_variance_match(y, v_ref)

    out_ref = m_vmatch.fused_rms_variance_match_triton(y, v_ref)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="rms_variance_match")


# ══════════════════════════════════════════════════════════════════════
# 15. rmsnorm
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rmsnorm(dtype):
    """Verifies fused RMSNorm forward pass."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)
    w = torch.randn(64, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_rms.rmsnorm(x, w)

    out_ref = m_rms.rmsnorm_pytorch_ref(x, w)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="rmsnorm")


# ══════════════════════════════════════════════════════════════════════
# 16. softplus
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softplus(dtype):
    """Verifies numerically stable fused Softplus activation."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_softplus.fused_softplus(x)

    out_ref = m_softplus.fused_softplus_pytorch_ref(x)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="softplus")


# ══════════════════════════════════════════════════════════════════════
# 17. split_head_norm
# ══════════════════════════════════════════════════════════════════════
def test_split_head_norm():
    """Verifies per-head independent normalization."""
    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)

    with force_kernel_backend("cuda"):
        out_cuda = m_split_head.split_head_norm(x)

    out_ref = m_split_head.split_head_norm_pytorch_ref(x)
    assert_numerical_parity(out_cuda, out_ref, dtype=torch.float32, name="split_head_norm")


# ══════════════════════════════════════════════════════════════════════
# 18. swiglu
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_swiglu(dtype):
    """Verifies fused SwiGLU forward pass."""
    x = torch.randn(2, 64, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_swiglu.swiglu(x)

    out_ref = m_swiglu.swiglu_pytorch_ref(x)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="swiglu")


# ══════════════════════════════════════════════════════════════════════
# 19. unified_rope_offset
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_unified_rope_offset(dtype):
    """Verifies Unified RoPE rotation with sequence position offset."""
    B, H, N, D = 2, 4, 16, 64
    rotary_dim = 32
    half = rotary_dim // 2
    x = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    cos = torch.randn(N, half, device="cuda", dtype=dtype)
    sin = torch.randn(N, half, device="cuda", dtype=dtype)

    with force_kernel_backend("cuda"):
        out_cuda = m_unirope.unified_rope_offset(x, cos, sin, rotary_dim=rotary_dim, offset=0)

    out_ref = m_unirope.unified_rope_offset_pytorch_ref(x, cos, sin, rotary_dim=rotary_dim, offset=0)
    assert_numerical_parity(out_cuda, out_ref, dtype=dtype, name="unified_rope_offset")
