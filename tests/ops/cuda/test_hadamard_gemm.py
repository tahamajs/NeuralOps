# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.hadamard_gemm import (
    hadamard_w4a16_gemm,
    hadamard_w4a16_gemm_pytorch_ref,
)


def _make_tensors(M: int, K: int, N: int, group_size: int = 32, with_bias: bool = True, batch_dims: tuple[int, ...] = ()):
    num_groups = K // group_size
    shape = (*batch_dims, M, K) if batch_dims else (M, K)
    x = torch.randn(*shape, dtype=torch.float32)
    w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    scales = torch.randn(N, num_groups, dtype=torch.float32)
    zeros = torch.zeros(N, num_groups, dtype=torch.float32)
    bias = torch.randn(N, dtype=torch.float32) if with_bias else None
    return x, w_packed, scales, zeros, bias


def test_hadamard_gemm_symbols():
    """Verify export symbols on module and top-level package."""
    import importlib
    mod = importlib.import_module("neuralops.ops.cuda.hadamard_gemm")

    assert hasattr(mod, "hadamard_w4a16_gemm")
    assert hasattr(mod, "hadamard_w4a16_gemm_pytorch_ref")
    assert "hadamard_w4a16_gemm" in mod.__all__
    assert "hadamard_w4a16_gemm_pytorch_ref" in mod.__all__

    # Direct import from package
    from neuralops.ops.cuda.hadamard_gemm import (
        hadamard_w4a16_gemm as fn1,
        hadamard_w4a16_gemm_pytorch_ref as fn2,
    )
    assert callable(fn1)
    assert callable(fn2)

    # Top-level lazy dispatch via noc
    assert callable(noc.hadamard_w4a16_gemm)
    assert callable(noc.hadamard_w4a16_gemm_pytorch_ref)
    assert callable(noc.hadamard_gemm)
    assert noc.hadamard_gemm is noc.hadamard_w4a16_gemm


def test_hadamard_gemm_pytorch_ref_2d():
    """Verify PyTorch reference implementation with 2D inputs."""
    M, K, N = 4, 64, 32
    group_size = 32
    x, w_packed, scales, zeros, bias = _make_tensors(M, K, N, group_size=group_size, with_bias=True)

    out = hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    assert out.shape == (M, N)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # Without bias
    out_nobias = hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size, bias=None)
    assert out_nobias.shape == (M, N)
    assert not torch.isnan(out_nobias).any()
    assert not torch.isinf(out_nobias).any()


def test_hadamard_gemm_pytorch_ref_multidim():
    """Verify PyTorch reference implementation with multidimensional batch dimensions."""
    B, S, K, N = 2, 8, 64, 16
    group_size = 32
    x, w_packed, scales, zeros, bias = _make_tensors(S, K, N, group_size=group_size, with_bias=True, batch_dims=(B,))

    out = hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    assert out.shape == (B, S, N)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hadamard_gemm_cpu_fallback_parity():
    """Verify that dispatched hadamard_w4a16_gemm matches PyTorch reference on CPU."""
    M, K, N = 8, 128, 64
    group_size = 32
    x, w_packed, scales, zeros, bias = _make_tensors(M, K, N, group_size=group_size, with_bias=True)

    ref = hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    out = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    top_out = noc.hadamard_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)

    assert out.shape == (M, N)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out, top_out, atol=1e-5, rtol=1e-5)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hadamard_gemm_varying_group_sizes():
    """Verify correct behavior with different group sizes."""
    M, K, N = 4, 64, 16
    for group_size in (16, 32, 64):
        x, w_packed, scales, zeros, bias = _make_tensors(M, K, N, group_size=group_size, with_bias=True)
        out = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
        assert out.shape == (M, N)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_hadamard_gemm_autograd_flow():
    """Verify backward gradient propagation for x and bias."""
    M, K, N = 4, 64, 32
    group_size = 32
    x, w_packed, scales, zeros, bias = _make_tensors(M, K, N, group_size=group_size, with_bias=True)

    x.requires_grad_(True)
    bias.requires_grad_(True)

    out = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

    assert bias.grad is not None
    assert bias.grad.shape == bias.shape
    assert not torch.isnan(bias.grad).any()
    assert not torch.isinf(bias.grad).any()


def test_hadamard_gemm_autograd_with_hadamard_transform():
    """Verify autograd with apply_hadamard=True."""
    M, K, N = 2, 64, 32
    group_size = 32
    x, w_packed, scales, zeros, bias = _make_tensors(M, K, N, group_size=group_size, with_bias=False)

    x.requires_grad_(True)
    out = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=None, apply_hadamard=True)
    loss = (out ** 2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hadamard_gemm_cuda_execution():
    """Verify CUDA kernel execution and parity when CUDA device is present."""
    device = torch.device("cuda")
    M, K, N = 8, 64, 32
    group_size = 32
    num_groups = K // group_size

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    scales = torch.randn(N, num_groups, dtype=torch.float32, device=device)
    zeros = torch.zeros(N, num_groups, dtype=torch.float32, device=device)
    bias = torch.randn(N, dtype=torch.float16, device=device)

    out = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)
    assert out.is_cuda
    assert out.shape == (M, N)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hadamard_gemm_agents_compliance():
    """Verify AGENTS.md rules compliance across all files in hadamard_gemm directory."""
    base = Path("neuralops/ops/cuda/hadamard_gemm")
    files = list(base.glob("*.*"))
    assert len(files) > 0, "No files found in hadamard_gemm directory"

    for p in files:
        if p.suffix in (".py", ".cpp", ".cu", ".cuh", ".h"):
            text = p.read_text(encoding="utf-8")
            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"torch.cuda.amp found in {p}"
            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"time.time() found in {p}"
            # Rule 2.5 & Anti-patterns: No hardcoded cuda:0
            assert "cuda:0" not in text, f"Hardcoded cuda:0 found in {p}"
            # Rule 6.3: File size <= 800 lines
            line_count = len(text.splitlines())
            assert line_count <= 800, f"{p} exceeds 800 lines ({line_count} lines)"
