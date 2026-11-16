# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from pathlib import Path
import pytest
import torch
from torch.amp import autocast
import neuralops.ops.cuda as noc
from neuralops.ops.cuda.differential_attention import differential_attention

def test_differential_attention_cpu_fallback():
    B, H, L, D = 2, 4, 16, 32
    q1 = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    q2 = torch.randn(B, H, L, D)
    k2 = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    out = differential_attention(q1, k1, q2, k2, v, lambda_val=0.8)
    assert out.shape == (B, H, L, D)
    assert not torch.isnan(out).any()

def test_differential_attention_top_level_dispatch():
    B, H, L, D = 1, 2, 8, 16
    q1 = torch.randn(B, H, L, D)
    k1 = torch.randn(B, H, L, D)
    q2 = torch.randn(B, H, L, D)
    k2 = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    out = noc.differential_attention(q1, k1, q2, k2, v)
    assert out.shape == (B, H, L, D)

def test_differential_attention_compliance():
    base = Path("neuralops/ops/cuda/differential_attention")
    for p in base.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert len(text.splitlines()) <= 800
