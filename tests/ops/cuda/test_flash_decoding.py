# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.flash_decoding import flash_decoding_attention

def test_flash_decoding_cpu_fallback():
    q = torch.randn(2, 4, 1, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    out = flash_decoding_attention(q, k, v)
    assert out.shape == (2, 4, 1, 32)
    assert not torch.isnan(out).any()

def test_flash_decoding_top_level_dispatch():
    q = torch.randn(1, 2, 1, 16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)
    out = noc.flash_decoding(q, k, v)
    assert out.shape == (1, 2, 1, 16)

def test_flash_decoding_compliance():
    base = Path("neuralops/ops/cuda/flash_decoding")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
