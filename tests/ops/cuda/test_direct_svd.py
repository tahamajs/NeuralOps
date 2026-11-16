# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.direct_svd import direct_svd_compress

def test_direct_svd_cpu_fallback():
    W = torch.randn(64, 64)
    out = direct_svd_compress(W)
    assert out.shape == W.shape
    assert not torch.isnan(out).any()

def test_direct_svd_top_level_dispatch():
    W = torch.randn(32, 32)
    out = noc.direct_svd(W)
    assert out.shape == W.shape

def test_direct_svd_compliance():
    base = Path("neuralops/ops/cuda/direct_svd")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
