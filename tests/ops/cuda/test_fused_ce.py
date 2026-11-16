# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fused_ce import fused_cross_entropy

def test_fused_ce_cpu_fallback():
    logits = torch.randn(4, 10)
    targets = torch.tensor([1, 0, 3, 2])
    out = fused_cross_entropy(logits, targets)
    assert out.numel() == 1
    assert not torch.isnan(out).any()

def test_fused_ce_top_level_dispatch():
    logits = torch.randn(2, 5)
    targets = torch.tensor([0, 1])
    out = noc.fused_ce(logits, targets)
    assert out.numel() == 1

def test_fused_ce_compliance():
    base = Path("neuralops/ops/cuda/fused_ce")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
