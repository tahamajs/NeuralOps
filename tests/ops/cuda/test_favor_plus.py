# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.favor_plus import favor_plus

def test_favor_plus_cpu_fallback():
    proj = torch.randn(64, 32)
    diag_sq = torch.randn(64)
    out = favor_plus(proj, diag_sq, norm_factor=1.0)
    assert out.shape == proj.shape
    assert not torch.isnan(out).any()

def test_favor_plus_top_level_dispatch():
    proj = torch.randn(32, 16)
    diag_sq = torch.randn(32)
    out = noc.favor_plus(proj, diag_sq, norm_factor=1.0)
    assert out.shape == proj.shape

def test_favor_plus_compliance():
    base = Path("neuralops/ops/cuda/favor_plus")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
