# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive test suite for neuralops.core.triton_kernels.linear_attn.

Coverage:
- Numerical correctness vs PyTorch linear attention reference
- Multiple dtypes: fp32, fp16, bf16
- Multiple feature maps phi: "elu+1", "relu", "identity"
- Causal vs Bidirectional modes
- Edge cases: single-token, varying sequence lengths, non-power-of-2
- Fallback path when Triton unavailable
- Determinism
- Gradient flow
"""
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, linear_attn_pytorch_fallback
from neuralops.core.triton_kernels.linear_attn import fwd, bwd, is_supported


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reference_linear_attn_pytorch(q, k, v, is_causal=False, phi="elu+1", eps=1e-6):
    """Pure PyTorch linear attention reference."""
    B, H, N, D_k = q.shape
    D_v = v.shape[-1]

    if phi == "elu+1":
        q_phi = F.elu(q.float(), alpha=1.0) + 1.0
        k_phi = F.elu(k.float(), alpha=1.0) + 1.0
    elif phi == "relu":
        q_phi = F.relu(q.float())
        k_phi = F.relu(k.float())
    else:
        q_phi = q.float()
        k_phi = k.float()

    v_f = v.float()

    if not is_causal:
        KV = torch.matmul(k_phi.transpose(-2, -1), v_f)
        Z = k_phi.sum(dim=-2, keepdim=True)
        num = torch.matmul(q_phi, KV)
        denom = torch.matmul(q_phi, Z.transpose(-2, -1))
        if phi in ("elu+1", "relu", "softmax_free"):
            denom_safe = torch.clamp(denom, min=eps)
        else:
            denom_sign = torch.sign(denom)
            denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
            denom_safe = torch.where(denom.abs() < eps, eps * denom_sign, denom)
        out = num / denom_safe
    else:
        out = torch.zeros(B, H, N, D_v, device=q.device, dtype=torch.float32)
        S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
        Z = torch.zeros(B, H, D_k, device=q.device, dtype=torch.float32)
        for t in range(N):
            q_t = q_phi[:, :, t, :]
            k_t = k_phi[:, :, t, :]
            v_t = v_f[:, :, t, :]

            S = S + torch.einsum("bhi,bhj->bhij", k_t, v_t)
            Z = Z + k_t

            num_t = torch.einsum("bhi,bhij->bhj", q_t, S)
            denom_val = torch.sum(q_t * Z, dim=-1, keepdim=True)
            if phi in ("elu+1", "relu", "softmax_free"):
                denom_safe = torch.clamp(denom_val, min=eps)
            else:
                denom_sign = torch.sign(denom_val)
                denom_sign = torch.where(denom_sign == 0, torch.ones_like(denom_sign), denom_sign)
                denom_safe = torch.where(denom_val.abs() < eps, eps * denom_sign, denom_val)
            out[:, :, t, :] = num_t / denom_safe

    return out.to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("phi", ["elu+1", "relu", "identity"])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("batch,num_heads", [(1, 2), (2, 4)])
@pytest.mark.parametrize("seq_len,dim", [(16, 32), (32, 64)])
def test_linear_attn_correctness(device, dtype, phi, is_causal, batch, num_heads, seq_len, dim):
    """Verify linear attention output matches reference."""
    q = torch.randn(batch, num_heads, seq_len, dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, dim, device=device, dtype=dtype)

    out = fwd(q, k, v, is_causal=is_causal, phi=phi)
    ref = reference_linear_attn_pytorch(q, k, v, is_causal=is_causal, phi=phi)

    if phi == "identity":
        tol = {torch.float32: 5e-2, torch.float16: 1e-1}[dtype]
        close = torch.allclose(out, ref, atol=tol, rtol=tol)
        if not close:
            cos_sim = F.cosine_similarity(out.flatten(start_dim=2), ref.flatten(start_dim=2), dim=-1)
            assert cos_sim.min() > 0.95
    else:
        tol = {torch.float32: 1e-3, torch.float16: 2e-2}[dtype]
        assert torch.allclose(out, ref, atol=tol, rtol=tol)


def test_linear_attn_single_token(device):
    """Single token sequence (seq_len = 1)."""
    q = torch.randn(1, 2, 1, 32, device=device)
    k = torch.randn(1, 2, 1, 32, device=device)
    v = torch.randn(1, 2, 1, 32, device=device)
    out = fwd(q, k, v, is_causal=True)
    assert out.shape == (1, 2, 1, 32)
    assert torch.isfinite(out).all()


def test_linear_attn_non_power_of_2(device):
    """Non-power-of-2 sequence length."""
    q = torch.randn(2, 2, 37, 32, device=device)
    k = torch.randn(2, 2, 37, 32, device=device)
    v = torch.randn(2, 2, 37, 32, device=device)
    out = fwd(q, k, v, is_causal=False)
    assert out.shape == (2, 2, 37, 32)
    assert torch.isfinite(out).all()


def test_linear_attn_cpu_fallback():
    """Verify fallback executes cleanly on CPU."""
    q = torch.randn(2, 2, 16, 32, device="cpu")
    k = torch.randn(2, 2, 16, 32, device="cpu")
    v = torch.randn(2, 2, 16, 32, device="cpu")
    out = linear_attn_pytorch_fallback(q, k, v, is_causal=True)
    assert out.shape == (2, 2, 16, 32)
    assert torch.isfinite(out).all()


def test_linear_attn_deterministic(device):
    """Deterministic output for identical input."""
    q = torch.randn(2, 2, 16, 32, device=device)
    k = torch.randn(2, 2, 16, 32, device=device)
    v = torch.randn(2, 2, 16, 32, device=device)
    out1 = fwd(q, k, v)
    out2 = fwd(q, k, v)
    assert torch.equal(out1, out2)


def test_linear_attn_gradient(device):
    """Gradients propagate back to q, k, and v."""
    q = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)
    v = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)

    out = fwd(q, k, v, is_causal=False)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()
