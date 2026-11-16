# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.entmax_bisect import entmax_bisect

def test_entmax_bisect_cpu_fallback():
    scores = torch.randn(2, 16, 16)
    out = entmax_bisect(scores)
    assert out.shape == scores.shape
    assert not torch.isnan(out).any()

def test_entmax_bisect_top_level_dispatch():
    scores = torch.randn(1, 8, 8)
    out = noc.entmax_bisect(scores)
    assert out.shape == scores.shape

def test_entmax_bisect_compliance():
    base = Path("neuralops/ops/cuda/entmax_bisect")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
