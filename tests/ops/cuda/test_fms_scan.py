# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.fms_scan import fms_scan

def test_fms_scan_cpu_fallback():
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    decay = torch.randn(4)
    out = fms_scan(q, k, v, decay)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()

def test_fms_scan_top_level_dispatch():
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    decay = torch.randn(2)
    out = noc.fms_scan(q, k, v, decay)
    assert out.shape == q.shape

def test_fms_scan_compliance():
    base = Path("neuralops/ops/cuda/fms_scan")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
