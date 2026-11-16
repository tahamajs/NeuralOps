# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Hadamard Linear Attention (HLA) CUDA/C++ kernel and fallback dispatch (Wave 11 W11-K18).

Verifies:
1. Operator importability and symbol resolution from neuralops.ops.cuda.hla_attention and neuralops.ops.cuda.
2. Execution through PyTorch reference fallback on CPU / non-CUDA environments.
3. Shape contracts, decay tensor dimensionality handling, and numerical stability (no NaNs, no Infs).
4. Numerical parity between dispatch interface and PyTorch reference implementation.
5. Autograd gradient propagation through q, k, and v.
6. AMP (torch.amp.autocast) compatibility on CPU.
7. Architectural and code quality compliance with AGENTS.md rules.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from torch.amp import autocast

import neuralops.ops.cuda as noc
from neuralops.ops.cuda import force_kernel_backend, list_cuda_kernels
from neuralops.ops.cuda.hla_attention import (
    hla_attention,
    hla_attention_pytorch_ref,
    hla_attention_ref,
)


def test_hla_attention_symbols_and_exports() -> None:
    """Verify operator callability and exports across package interfaces."""
    assert callable(hla_attention)
    assert callable(hla_attention_ref)
    assert callable(hla_attention_pytorch_ref)
    assert hla_attention_pytorch_ref is hla_attention_ref

    # Top-level lazy dispatch resolution
    assert hasattr(noc, "hla_attention")
    assert callable(noc.hla_attention)
    assert hasattr(noc, "hla_attention_ref")
    assert callable(noc.hla_attention_ref)
    assert hasattr(noc, "hla_attention_pytorch_ref")
    assert callable(noc.hla_attention_pytorch_ref)

    # Catalog registration
    modules = list_cuda_kernels(category="modules")
    assert "hla_attention" in modules


def test_hla_attention_cpu_fallback_forward() -> None:
    """Verify execution through PyTorch CPU fallback produces valid finite output."""
    B, H, L, D = 2, 4, 16, 32
    torch.manual_seed(42)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(B, H, L))

    out = hla_attention(q, k, v, decay)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_hla_attention_dtypes(dtype: torch.dtype) -> None:
    """Verify output preserves query tensor dtype."""
    B, H, L, D = 1, 2, 8, 16
    torch.manual_seed(101)
    q = torch.rand(B, H, L, D, dtype=dtype) + 0.1
    k = torch.rand(B, H, L, D, dtype=dtype) + 0.1
    v = torch.randn(B, H, L, D, dtype=dtype)
    decay = torch.sigmoid(torch.randn(B, H, L, dtype=dtype))

    out = hla_attention(q, k, v, decay)

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize(
    "decay_shape",
    [
        (2, 4, 16),  # Standard 3D (B, H, L)
        (2, 4, 16, 1),  # 4D with trailing singleton
        (4, 16),  # 2D per-head sequence decay (H, L)
        (4,),  # 1D static per-head decay (H,)
        (1, 4, 16),  # Broadcastable batch
        (2, 1, 16),  # Broadcastable head
    ],
)
def test_hla_attention_decay_shapes(decay_shape: tuple[int, ...]) -> None:
    """Verify decay tensor broadcastability across varying dimension layouts."""
    B, H, L, D = 2, 4, 16, 16
    torch.manual_seed(202)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(*decay_shape))

    out = hla_attention(q, k, v, decay)
    ref = hla_attention_ref(q, k, v, decay)

    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_hla_attention_numerical_parity_with_ref() -> None:
    """Verify numerical equivalence between dispatch interface and PyTorch reference."""
    B, H, L, D = 2, 3, 12, 32
    torch.manual_seed(303)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(B, H, L))

    with force_kernel_backend("pytorch"):
        out_dispatched = hla_attention(q, k, v, decay)

    out_ref = hla_attention_ref(q, k, v, decay)
    out_noc = noc.hla_attention(q, k, v, decay)

    assert torch.allclose(out_dispatched, out_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_noc, out_ref, atol=1e-6, rtol=1e-5)


def test_hla_attention_autograd_gradients() -> None:
    """Verify backward gradient flow through q, k, and v tensors."""
    B, H, L, D = 2, 2, 6, 16
    torch.manual_seed(404)
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    decay = torch.sigmoid(torch.randn(B, H, L))

    out = hla_attention(q, k, v, decay)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


def test_hla_attention_decay_influence() -> None:
    """Verify that different decay values produce distinct output representations."""
    B, H, L, D = 1, 2, 8, 16
    torch.manual_seed(505)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)

    decay_low = torch.full((B, H, L), 0.1)
    decay_high = torch.full((B, H, L), 0.99)

    out_low = hla_attention(q, k, v, decay_low)
    out_high = hla_attention(q, k, v, decay_high)

    assert not torch.allclose(out_low, out_high)


def test_hla_attention_eps_stability() -> None:
    """Verify numerical stability with varying epsilon configurations."""
    B, H, L, D = 1, 2, 4, 8
    torch.manual_seed(606)
    q = torch.zeros(B, H, L, D)  # Zero query tests denom stability
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(B, H, L))

    out = hla_attention(q, k, v, decay, eps=1e-4)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_hla_attention_torch_amp_compliance() -> None:
    """Verify execution under torch.amp.autocast without deprecations or errors."""
    B, H, L, D = 1, 2, 8, 16
    torch.manual_seed(707)
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    decay = torch.sigmoid(torch.randn(B, H, L))

    with autocast(device_type="cpu", dtype=torch.bfloat16):
        out = hla_attention(q, k, v, decay)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_hla_attention_agents_md_compliance() -> None:
    """Verify static architectural rules defined in AGENTS.md."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent / "neuralops" / "ops" / "cuda" / "hla_attention"
    assert base_dir.is_dir(), f"Directory not found: {base_dir}"

    for p in base_dir.glob("*"):
        if p.is_file() and p.suffix in (".py", ".cu", ".cpp"):
            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()

            # Rule 6.3: Max 800 lines per file
            assert len(lines) <= 800, f"{p.name} exceeds 800 lines ({len(lines)})"

            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{p.name} contains deprecated torch.cuda.amp"

            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{p.name} contains time.time()"

            # Rule 2.5: No hardcoded cuda:0
            assert "cuda:0" not in text, f"{p.name} contains hardcoded cuda:0"

            # Rule 1.4: No def main() in library files
            assert "def main()" not in text, f"{p.name} contains def main()"
