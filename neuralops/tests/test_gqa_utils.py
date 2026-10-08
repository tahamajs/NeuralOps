"""Tests for repeat_kv_optimized."""
import torch
from neuralops.core.gqa_utils import repeat_kv, repeat_kv_optimized


def test_repeat_kv_optimized():
    x = torch.randn(2, 4, 16, 32)
    rep = 2
    out_orig = repeat_kv(x, rep)
    out_opt = repeat_kv_optimized(x, rep)
    assert torch.allclose(out_orig, out_opt)
    assert out_opt.shape == (2, 8, 16, 32)


def test_repeat_kv_optimized_identity():
    x = torch.randn(2, 4, 16, 32)
    out_opt = repeat_kv_optimized(x, 1)
    assert out_opt is x or torch.equal(out_opt, x)
