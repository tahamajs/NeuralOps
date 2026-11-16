# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.entropy_pruning import entropy_pruning

def test_entropy_pruning_cpu_fallback():
    scores = torch.randn(64, 64)
    out = entropy_pruning(scores)
    assert out.shape == (64,)
    assert not torch.isnan(out).any()

def test_entropy_pruning_top_level_dispatch():
    scores = torch.randn(32, 32)
    out = noc.entropy_pruning(scores)
    assert out.shape == (32,)

def test_entropy_pruning_compliance():
    base = Path("neuralops/ops/cuda/entropy_pruning")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
