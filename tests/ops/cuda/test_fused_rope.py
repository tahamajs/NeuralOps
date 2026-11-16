# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Fused RoPE (Rotary Position Embedding) CUDA kernel and fallback dispatch (Wave 11 W11-K2).

Verifies:
1. Operator importability and symbol resolution from neuralops.ops.cuda.fused_rope and top-level neuralops.ops.cuda.
2. Execution through PyTorch reference fallback on CPU / non-CUDA environments.
3. Shape contracts, varying dimension broadcasting, and numerical stability (no NaNs, no Infs).
4. Numerical parity between dispatch interface and PyTorch reference.
5. Mathematical identity and rotation correctness under known angles.
6. Autograd backward gradient propagation to input tensor x.
7. Support across multiple precisions (float32, float16, bfloat16).
8. torch.amp.autocast compatibility on CPU.
9. Compliance with AGENTS.md rules (line count <= 800, no deprecated torch.cuda.amp, no time.time() for GPU timing,
   no hardcoded cuda:0, no main() in library files).
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fused_rope import (
    fused_rope,
    fused_rope_pytorch_ref,
    fused_rotary_emb,
    fused_rotary_emb_pytorch_ref,
)


def test_fused_rope_import_and_symbols() -> None:
    """Verify top-level and package exports for fused_rope and fused_rotary_emb."""
    assert callable(fused_rope)
    assert callable(fused_rope_pytorch_ref)
    assert callable(fused_rotary_emb)
    assert callable(fused_rotary_emb_pytorch_ref)

    assert hasattr(noc, "fused_rope")
    assert callable(noc.fused_rope)
    assert hasattr(noc, "fused_rotary_emb")
    assert callable(noc.fused_rotary_emb)
    assert hasattr(noc, "fused_rope_pytorch_ref")
    assert callable(noc.fused_rope_pytorch_ref)
    assert hasattr(noc, "fused_rotary_emb_pytorch_ref")
    assert callable(noc.fused_rotary_emb_pytorch_ref)

    modules = noc.list_cuda_kernels("modules")
    assert "fused_rope" in modules
    assert "fused_rope" in noc.CUDA_KERNEL_CATALOG


def test_fused_rope_cpu_fallback() -> None:
    """Verify execution on CPU via fallback dispatch with no NaNs or Infs."""
    B, H, L, D = 2, 4, 16, 64
    x = torch.randn(B, H, L, D)
    half_d = D // 2
    cos = torch.cos(torch.randn(L, half_d))
    sin = torch.sin(torch.randn(L, half_d))

    out = fused_rope(x, cos, sin)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_fused_rope_top_level_dispatch() -> None:
    """Verify top-level noc.fused_rope and noc.fused_rotary_emb dispatch."""
    B, H, L, D = 1, 2, 8, 32
    x = torch.randn(B, H, L, D)
    half_d = D // 2
    cos = torch.cos(torch.randn(L, half_d))
    sin = torch.sin(torch.randn(L, half_d))

    out1 = noc.fused_rope(x, cos, sin)
    out2 = noc.fused_rotary_emb(x, cos, sin)

    assert out1.shape == x.shape
    assert out2.shape == x.shape
    assert torch.equal(out1, out2)
    assert not torch.isnan(out1).any()


def test_fused_rope_numerical_parity() -> None:
    """Verify exact equivalence between dispatch output and pure reference implementation."""
    B, H, L, D = 2, 8, 32, 64
    torch.manual_seed(42)
    x = torch.randn(B, H, L, D)
    half_d = D // 2
    cos = torch.cos(torch.randn(L, half_d))
    sin = torch.sin(torch.randn(L, half_d))

    ref = fused_rotary_emb_pytorch_ref(x, cos, sin)
    out = fused_rotary_emb(x, cos, sin)
    ref_alias = fused_rope_pytorch_ref(x, cos, sin)
    out_alias = fused_rope(x, cos, sin)

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_alias, ref_alias, atol=1e-6, rtol=1e-5)


def test_fused_rope_mathematical_identity() -> None:
    """Verify mathematical properties for identity rotation (cos=1, sin=0) and 90-deg rotation."""
    B, H, L, D = 2, 4, 8, 16
    x = torch.randn(B, H, L, D)
    half_d = D // 2

    # Identity rotation: cos = 1, sin = 0 -> output should equal x
    cos_ident = torch.ones(L, half_d)
    sin_ident = torch.zeros(L, half_d)
    out_ident = fused_rope(x, cos_ident, sin_ident)
    assert torch.allclose(out_ident, x, atol=1e-6)

    # 90-degree rotation: cos = 0, sin = 1 -> o1 = -x2, o2 = x1
    cos_90 = torch.zeros(L, half_d)
    sin_90 = torch.ones(L, half_d)
    out_90 = fused_rope(x, cos_90, sin_90)
    expected_o1 = -x[..., half_d:]
    expected_o2 = x[..., :half_d]
    expected_90 = torch.cat([expected_o1, expected_o2], dim=-1)
    assert torch.allclose(out_90, expected_90, atol=1e-6)


def test_fused_rope_varying_shapes_and_broadcasting() -> None:
    """Verify execution with varying dimensionalities and broadcasted cos/sin shapes."""
    torch.manual_seed(99)

    # 4D x with 2D cos/sin
    x4d = torch.randn(2, 4, 16, 64)
    cos2d = torch.cos(torch.randn(16, 32))
    sin2d = torch.sin(torch.randn(16, 32))
    out4d = fused_rope(x4d, cos2d, sin2d)
    assert out4d.shape == (2, 4, 16, 64)
    assert not torch.isnan(out4d).any()

    # 4D x with 4D cos/sin
    cos4d = torch.cos(torch.randn(2, 4, 16, 32))
    sin4d = torch.sin(torch.randn(2, 4, 16, 32))
    out4d_full = fused_rope(x4d, cos4d, sin4d)
    assert out4d_full.shape == (2, 4, 16, 64)
    assert not torch.isnan(out4d_full).any()

    # 3D x with 2D cos/sin
    x3d = torch.randn(2, 16, 64)
    out3d = fused_rope(x3d, cos2d, sin2d)
    assert out3d.shape == (2, 16, 64)
    assert not torch.isnan(out3d).any()


def test_fused_rope_autograd_gradients() -> None:
    """Verify backward autograd pass propagates gradients correctly to input x."""
    B, H, L, D = 2, 4, 8, 32
    torch.manual_seed(7)
    x = torch.randn(B, H, L, D, requires_grad=True)
    half_d = D // 2
    cos = torch.cos(torch.randn(L, half_d))
    sin = torch.sin(torch.randn(L, half_d))

    out = fused_rope(x, cos, sin)
    loss = (out**2).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_fused_rope_dtypes() -> None:
    """Verify execution across float32, float16, and bfloat16 data types."""
    B, H, L, D = 1, 2, 8, 16
    half_d = D // 2

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(B, H, L, D, dtype=dtype)
        cos = torch.cos(torch.randn(L, half_d, dtype=dtype))
        sin = torch.sin(torch.randn(L, half_d, dtype=dtype))

        out = fused_rope(x, cos, sin)
        assert out.dtype == dtype
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


def test_fused_rope_amp_autocast_cpu() -> None:
    """Verify execution under torch.amp.autocast on CPU without error."""
    B, H, L, D = 2, 2, 8, 32
    x = torch.randn(B, H, L, D)
    half_d = D // 2
    cos = torch.cos(torch.randn(L, half_d))
    sin = torch.sin(torch.randn(L, half_d))

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = fused_rope(x, cos, sin)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_fused_rope_compliance() -> None:
    """Verify strict AGENTS.md rules compliance for fused_rope module."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent / "neuralops" / "ops" / "cuda" / "fused_rope"
    assert base_dir.is_dir(), f"Directory not found: {base_dir}"

    for p in base_dir.iterdir():
        if p.suffix in (".py", ".cu", ".cpp", ".cuh", ".h"):
            text = p.read_text(encoding="utf-8")
            # Rule 6.3: Line count <= 800
            assert len(text.splitlines()) <= 800, f"{p.name} exceeds 800 lines"
            # Rule 2.1: No deprecated torch.cuda.amp
            assert "torch.cuda.amp" not in text, f"{p.name} contains deprecated torch.cuda.amp"
            # Rule 2.2: No time.time() for GPU timing
            assert "time.time()" not in text, f"{p.name} uses time.time()"
            # No hardcoded cuda:0
            assert "cuda:0" not in text, f"{p.name} hardcodes cuda:0"
            # Rule 1.4: No def main() in library files
            assert "def main()" not in text, f"{p.name} defines main()"
            assert '__name__ == "__main__"' not in text, f"{p.name} has main execution block"
