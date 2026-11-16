# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fp8_gemm import fp8_gemm

def test_fp8_gemm_cpu_fallback():
    x = torch.randn(2, 32)
    w = torch.randn(32, 16)
    out = fp8_gemm(x, w)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()

def test_fp8_gemm_top_level_dispatch():
    x = torch.randn(1, 16)
    w = torch.randn(16, 8)
    out = noc.fp8_gemm(x, w)
    assert out.shape == (1, 8)

def test_fp8_gemm_compliance():
    base = Path("neuralops/ops/cuda/fp8_gemm")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
