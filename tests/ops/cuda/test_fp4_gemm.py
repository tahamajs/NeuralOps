# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fp4_gemm import fp4_gemm

def test_fp4_gemm_cpu_fallback():
    x = torch.randn(2, 32)
    w_packed = torch.randint(0, 255, (16, 16), dtype=torch.uint8)
    scales = torch.ones(16)
    out = fp4_gemm(x, w_packed, scales)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

def test_fp4_gemm_top_level_dispatch():
    x = torch.randn(1, 16)
    w_packed = torch.randint(0, 255, (8, 8), dtype=torch.uint8)
    scales = torch.ones(8)
    out = noc.fp4_gemm(x, w_packed, scales)
    assert out.shape == (1, 8)

def test_fp4_gemm_compliance():
    base = Path("neuralops/ops/cuda/fp4_gemm")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
