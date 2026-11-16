# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.entropy_gate import spectral_entropy_svd_gate

def test_entropy_gate_cpu_fallback():
    x = torch.randn(2, 16, 32)
    V_r = torch.randn(32, 16)
    U_r = torch.randn(32, 16)
    out = spectral_entropy_svd_gate(x, V_r, U_r, effective_rank=16)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_entropy_gate_top_level_dispatch():
    x = torch.randn(1, 8, 16)
    V_r = torch.randn(16, 8)
    U_r = torch.randn(16, 8)
    out = noc.entropy_gate(x, V_r, U_r, effective_rank=8)
    assert out.shape == x.shape

def test_entropy_gate_compliance():
    base = Path("neuralops/ops/cuda/entropy_gate")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
