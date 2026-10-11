"""Tests for compression engine enhancements (GPTQ, AWQ, SmoothQuant, Wanda, LoRA/QLoRA/DoRA)."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.gptq import GPTQQuantizer
from neuralops.compression_engine.quantization.awq import AWQQuantizer
from neuralops.compression_engine.quantization.smoothquant import smoothquant_layer
from neuralops.compression_engine.pruning.wanda import wanda_prune_layer
from neuralops.compression_engine.low_rank.dora_qlora import (
    merge_lora_into_linear,
    extract_lora_from_diff,
    QLoRALinear,
    DoRALinear,
)


def test_gptq_quantizer():
    layer = nn.Linear(16, 16)
    q = GPTQQuantizer(layer, bits=4)
    x = torch.randn(8, 16)
    q.add_batch(x)
    res = q.quantize()
    assert res.shape == (16, 16)


def test_awq_quantizer():
    layer = nn.Linear(16, 16)
    awq = AWQQuantizer(layer, bits=4)
    x = torch.randn(8, 16)
    awq.observe(x)
    res = awq.quantize()
    assert res.shape == (16, 16)


def test_smoothquant():
    layer = nn.Linear(16, 16)
    act_scales = torch.ones(16) * 2.0
    layer_mod, scales = smoothquant_layer(layer, act_scales=act_scales)
    assert scales.shape == (16,)


def test_wanda_pruning():
    layer = nn.Linear(16, 16)
    act_norm = torch.ones(16)
    layer_mod, mask = wanda_prune_layer(layer, sparsity=0.5, act_norm=act_norm)
    assert mask.shape == (16, 16)
    assert torch.sum(layer.weight.data == 0) > 0


def test_lora_merge_and_extract():
    lin = nn.Linear(16, 16)
    orig_w = lin.weight.data.clone()
    A = torch.randn(4, 16)
    B = torch.randn(16, 4)
    merge_lora_into_linear(lin, A, B, scaling=1.0)
    assert not torch.equal(lin.weight.data, orig_w)

    diff = B @ A
    A_ext, B_ext = extract_lora_from_diff(diff, rank=4)
    assert torch.allclose(diff, B_ext @ A_ext, atol=1e-4)


def test_qlora_and_dora():
    x = torch.randn(2, 16)
    qlora = QLoRALinear(16, 16, rank=4)
    out_q = qlora(x)
    assert out_q.shape == (2, 16)

    dora = DoRALinear(16, 16, rank=4)
    out_d = dora(x)
    assert out_d.shape == (2, 16)
