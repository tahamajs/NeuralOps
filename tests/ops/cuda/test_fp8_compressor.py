# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fp8_compressor import fp8_compressor_compress

def test_fp8_compressor_cpu_fallback():
    W = torch.randn(64, 64)
    out = fp8_compressor_compress(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()

def test_fp8_compressor_top_level_dispatch():
    W = torch.randn(32, 32)
    out = noc.fp8_compressor(W)
    assert out.shape == W.shape

def test_fp8_compressor_compliance():
    base = Path("neuralops/ops/cuda/fp8_compressor")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
