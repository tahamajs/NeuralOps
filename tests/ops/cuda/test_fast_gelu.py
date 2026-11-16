# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fast_gelu import fast_gelu

def test_fast_gelu_cpu_fallback():
    x = torch.randn(2, 16, 32)
    out = fast_gelu(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_fast_gelu_top_level_dispatch():
    x = torch.randn(1, 8, 16)
    out = noc.fast_gelu(x)
    assert out.shape == x.shape

def test_fast_gelu_compliance():
    base = Path("neuralops/ops/cuda/fast_gelu")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
