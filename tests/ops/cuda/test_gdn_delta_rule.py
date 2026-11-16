# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path
import pytest
import torch

import neuralops.ops.cuda as noc
from neuralops.ops.cuda.gdn_delta_rule import gdn_delta_rule, gdn_delta_rule_pytorch_ref


def test_gdn_delta_rule_cpu_fallback():
    B, H, L, D = 2, 4, 8, 16
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))

    out = gdn_delta_rule(q, k, v, beta)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_gdn_delta_rule_top_level_dispatch():
    B, H, L, D = 1, 2, 4, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    beta = torch.sigmoid(torch.randn(B, H, L))

    out = noc.gdn_delta_rule(q, k, v, beta)
    ref = noc.gdn_delta_rule_pytorch_ref(q, k, v, beta)
    assert out.shape == q.shape
    assert ref.shape == q.shape
    assert torch.allclose(out, ref, atol=1e-5)


def test_gdn_delta_rule_autograd():
    B, H, L, D = 2, 2, 4, 8
    q = torch.randn(B, H, L, D, requires_grad=True)
    k = torch.randn(B, H, L, D, requires_grad=True)
    v = torch.randn(B, H, L, D, requires_grad=True)
    beta = torch.randn(B, H, L, requires_grad=True)

    out = gdn_delta_rule(q, k, v, beta)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert beta.grad is not None

    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
    assert not torch.isnan(beta.grad).any()

    assert not torch.isinf(q.grad).any()
    assert not torch.isinf(k.grad).any()
    assert not torch.isinf(v.grad).any()
    assert not torch.isinf(beta.grad).any()


def test_gdn_delta_rule_varying_shapes():
    shapes = [
        (1, 1, 1, 4),
        (2, 3, 5, 8),
        (1, 4, 16, 16),
    ]
    for B, H, L, D in shapes:
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        beta = torch.randn(B, H, L)
        out = gdn_delta_rule(q, k, v, beta)
        assert out.shape == (B, H, L, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_gdn_delta_rule_compliance():
    base = Path("neuralops/ops/cuda/gdn_delta_rule")
    py_files = list(base.glob("*.py"))
    assert len(py_files) > 0, "No python files found in kernel module directory"
    for p in py_files:
        text = p.read_text(encoding="utf-8")
        assert "torch.cuda.amp" not in text
        assert "time.time()" not in text
        assert "cuda:0" not in text
        assert len(text.splitlines()) <= 800
