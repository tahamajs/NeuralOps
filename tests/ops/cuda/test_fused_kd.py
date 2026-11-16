# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fused_kd import fused_kd_loss

def test_fused_kd_cpu_fallback():
    s = torch.randn(4, 10)
    t = torch.randn(4, 10)
    out = fused_kd_loss(s, t, temperature=2.0)
    assert out.numel() == 1
    assert not torch.isnan(out).any()

def test_fused_kd_top_level_dispatch():
    s = torch.randn(2, 5)
    t = torch.randn(2, 5)
    out = noc.fused_kd(s, t, temperature=2.0)
    assert out.numel() == 1

def test_fused_kd_compliance():
    base = Path("neuralops/ops/cuda/fused_kd")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
