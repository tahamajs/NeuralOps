# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.dynamic_conv1d import dynamic_conv1d

def test_dynamic_conv1d_cpu_fallback():
    v = torch.randn(2, 16, 32)
    w = torch.randn(32, 4)
    out = dynamic_conv1d(v, w)
    assert out.shape == (2, 32, 32)
    assert not torch.isnan(out).any()

def test_dynamic_conv1d_top_level_dispatch():
    v = torch.randn(1, 8, 16)
    w = torch.randn(16, 4)
    out = noc.dynamic_conv1d(v, w)
    assert out.shape == (1, 16, 16)

def test_dynamic_conv1d_compliance():
    base = Path("neuralops/ops/cuda/dynamic_conv1d")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
