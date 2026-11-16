# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.flash_decode_stage2 import flash_decode_stage2

def test_flash_decode_stage2_cpu_fallback():
    mid_out = torch.randn(2, 4, 16, 32)
    mid_lse = torch.randn(2, 4, 16)
    out = flash_decode_stage2(mid_out, mid_lse)
    assert out.shape == (2, 4, 1, 32)
    assert not torch.isnan(out).any()

def test_flash_decode_stage2_top_level_dispatch():
    mid_out = torch.randn(1, 2, 8, 16)
    mid_lse = torch.randn(1, 2, 8)
    out = noc.flash_decode_stage2(mid_out, mid_lse)
    assert out.shape == (1, 2, 1, 16)

def test_flash_decode_stage2_compliance():
    base = Path("neuralops/ops/cuda/flash_decode_stage2")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
