"""Tests for production LoRA merge/extraction, QLoRA, and DoRA."""
import torch
import torch.nn as nn
from neuralops.compression_engine.low_rank.dora_qlora import (
    merge_lora_into_linear,
    extract_lora_from_diff,
    QuantizedLinear,
    QLoRALinear,
    DoRALinear,
)


def test_lora_merge_real():
    lin = nn.Linear(32, 16, bias=False)
    orig_w = lin.weight.data.clone()
    A = torch.randn(4, 32)
    B = torch.randn(16, 4)
    res = merge_lora_into_linear(lin, A, B, scaling=0.5)
    assert res is lin
    assert torch.allclose(lin.weight.data, orig_w + 0.5 * (B @ A))


def test_lora_extract_real():
    delta_W = torch.randn(32, 16)
    A, B = extract_lora_from_diff(delta_W, rank=16)
    assert A.shape == (16, 16)
    assert B.shape == (32, 16)
    assert torch.allclose(delta_W, B @ A, atol=1e-4)


def test_qlora_real():
    qlora = QLoRALinear(32, 16, rank=4, bits=4)
    x = torch.randn(4, 32)
    out = qlora(x)
    assert out.shape == (4, 16)
    # Base weight shouldn't accumulate gradient
    out.sum().backward()
    assert qlora.base.weight.grad is None
    assert qlora.lora_A.grad is not None


def test_dora_real():
    dora = DoRALinear(32, 16, rank=4)
    x = torch.randn(4, 32)
    out = dora(x)
    assert out.shape == (4, 16)
    out.sum().backward()
    assert dora.base.weight.grad is None
    assert dora.magnitude.grad is not None
    assert dora.lora_A.grad is not None
