# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for Galerkin Attention CUDA kernel and CPU fallback.

Verifies:
1. Symbol exports, callability, and catalog integration in neuralops.ops.cuda.
2. Pure PyTorch CPU fallback execution and numerical sanity (finite, no NaNs/Infs).
3. Exact numerical parity between galerkin_attention and galerkin_attention_ref.
4. Mathematical invariants, scaling behavior with h_step, and zero preservation.
5. Multi-dimensional tensor shapes and edge case handling (sequence length 1).
6. Precision support: float32, float64, bfloat16, and torch.amp.autocast compliance (Rule 2.1).
7. Autograd backward pass and gradient flow through Level 3 fallback.
8. Backend routing and forced backend support via force_kernel_backend.
9. Full compliance with AGENTS.md rules (line count <= 800, no torch.cuda.amp, no time.time, no cuda:0).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
)
from neuralops.ops.cuda.galerkin_attention import (
    galerkin_attention,
    galerkin_attention_forward,
    galerkin_attention_pytorch_ref,
    galerkin_attention_ref,
)


def test_galerkin_attention_symbol_exports() -> None:
    """Verify symbol exports, aliases, callables, and catalog registration."""
    assert callable(galerkin_attention)
    assert callable(galerkin_attention_forward)
    assert callable(galerkin_attention_ref)
    assert callable(galerkin_attention_pytorch_ref)

    # Top-level lazy exports via neuralops.ops.cuda
    assert callable(noc.galerkin_attention)
    assert callable(noc.galerkin_attention_forward)
    assert callable(noc.galerkin_attention_ref)
    assert callable(noc.galerkin_attention_pytorch_ref)

    # Catalog presence
    assert "galerkin_attention" in noc.list_cuda_kernels("modules")
    assert "galerkin_attention" in noc.CUDA_KERNEL_CATALOG


def test_galerkin_attention_cpu_basic() -> None:
    """Verify basic forward execution on CPU produces valid shape and finite output."""
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 16, 32
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)

    out = galerkin_attention(q, k, v, h_step=0.01)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, H, N, D)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_galerkin_attention_top_level_dispatch() -> None:
    """Verify top-level dispatcher execution via neuralops.ops.cuda."""
    torch.manual_seed(101)
    B, H, N, D = 1, 2, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    out = noc.galerkin_attention(q, k, v)
    assert out.shape == (B, H, N, D)
    assert torch.isfinite(out).all()


def test_galerkin_attention_numerical_parity_with_reference() -> None:
    """Verify exact numerical parity between dispatcher and PyTorch reference."""
    torch.manual_seed(123)
    B, H, N, D = 2, 3, 24, 48
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)

    out = galerkin_attention(q, k, v, h_step=0.01)
    ref = galerkin_attention_ref(q, k, v, h_step=0.01)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_galerkin_attention_h_step_scaling() -> None:
    """Verify linear scaling with step size h_step."""
    torch.manual_seed(456)
    B, H, N, D = 1, 2, 12, 16
    q = torch.randn(B, H, N, D, dtype=torch.float32)
    k = torch.randn(B, H, N, D, dtype=torch.float32)
    v = torch.randn(B, H, N, D, dtype=torch.float32)

    out_1 = galerkin_attention(q, k, v, h_step=0.01)
    out_2 = galerkin_attention(q, k, v, h_step=0.02)
    ref_2 = galerkin_attention_ref(q, k, v, h_step=0.02)

    # 2x h_step must produce 2x output
    torch.testing.assert_close(out_2, out_1 * 2.0, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_2, ref_2, rtol=1e-5, atol=1e-5)


def test_galerkin_attention_zero_invariance() -> None:
    """Verify zero tensor invariants for queries, keys, and values."""
    B, H, N, D = 2, 2, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Zero queries -> zero output
    out_zero_q = galerkin_attention(torch.zeros_like(q), k, v)
    assert torch.all(out_zero_q == 0.0)

    # Zero keys -> zero output
    out_zero_k = galerkin_attention(q, torch.zeros_like(k), v)
    assert torch.all(out_zero_k == 0.0)

    # Zero values -> zero output
    out_zero_v = galerkin_attention(q, k, torch.zeros_like(v))
    assert torch.all(out_zero_v == 0.0)


@pytest.mark.parametrize(
    ("B", "H", "N", "D"),
    [
        (1, 1, 16, 32),
        (2, 4, 32, 64),
        (1, 8, 64, 32),
        (4, 2, 8, 16),
        (2, 2, 1, 16),  # Edge case: sequence length 1
    ],
)
def test_galerkin_attention_multi_dimensional_shapes(B: int, H: int, N: int, D: int) -> None:
    """Verify execution across diverse batch sizes, head counts, and sequence lengths."""
    torch.manual_seed(789)
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    out = galerkin_attention(q, k, v, h_step=0.005)
    ref = galerkin_attention_ref(q, k, v, h_step=0.005)

    assert out.shape == (B, H, N, D)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_galerkin_attention_precision_support(dtype: torch.dtype) -> None:
    """Verify execution across supported floating point precisions."""
    torch.manual_seed(999)
    B, H, N, D = 1, 2, 8, 16
    q = torch.randn(B, H, N, D, dtype=dtype)
    k = torch.randn(B, H, N, D, dtype=dtype)
    v = torch.randn(B, H, N, D, dtype=dtype)

    out = galerkin_attention(q, k, v, h_step=0.01)
    ref = galerkin_attention_ref(q, k, v, h_step=0.01)

    assert out.dtype == dtype
    assert out.shape == (B, H, N, D)
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_galerkin_attention_amp_autocast_compliance() -> None:
    """Verify compliance with Rule 2.1 (torch.amp.autocast, not torch.cuda.amp)."""
    B, H, N, D = 1, 2, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    with autocast("cpu", dtype=torch.bfloat16):
        out = galerkin_attention(q, k, v, h_step=0.01)

    assert out.shape == (B, H, N, D)
    assert torch.isfinite(out).all()


def test_galerkin_attention_autograd_backward() -> None:
    """Verify backward gradient computation and numerical parity on CPU fallback."""
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 8, 16
    q = torch.randn(B, H, N, D, requires_grad=True)
    k = torch.randn(B, H, N, D, requires_grad=True)
    v = torch.randn(B, H, N, D, requires_grad=True)

    out = galerkin_attention(q, k, v, h_step=0.01)
    loss = (out**2).sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()

    # Compare against reference analytical gradient
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = galerkin_attention_ref(q_ref, k_ref, v_ref, h_step=0.01)
    loss_ref = (out_ref**2).sum()
    loss_ref.backward()

    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-5, atol=1e-5)


def test_galerkin_attention_forced_backend_routing() -> None:
    """Verify backend forcing and telemetry recording."""
    torch.manual_seed(123)
    B, H, N, D = 1, 2, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Force pytorch backend
    with force_kernel_backend("pytorch"):
        out = galerkin_attention(q, k, v)
        assert out.shape == (B, H, N, D)
        assert get_last_dispatch_backend("galerkin_attention") == "pytorch"

    # Force cuda backend on non-CUDA host raises RuntimeError cleanly
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            with force_kernel_backend("cuda"):
                galerkin_attention(q, k, v)


def test_galerkin_attention_agents_md_compliance() -> None:
    """Verify strict compliance with AGENTS.md rules for galerkin_attention module."""
    base = Path("neuralops/ops/cuda/galerkin_attention")
    assert base.exists(), f"Directory {base} does not exist"

    for file_path in base.iterdir():
        if file_path.suffix in (".py", ".cu", ".cpp", ".cuh", ".h"):
            text = file_path.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Rule 6.3: File size <= 800 lines
            assert len(lines) <= 800, f"{file_path} exceeds 800 lines ({len(lines)})"

            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{file_path} contains forbidden torch.cuda.amp"

            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{file_path} contains forbidden time.time()"

            # Rule 10: No hardcoded cuda:0
            assert "cuda:0" not in text, f"{file_path} contains hardcoded cuda:0"
