#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Numerical verification tests for production GPTQ."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.gptq import GPTQQuantizer, quantize_model_gptq


def test_gptq_real_128x128():
    torch.manual_seed(42)
    layer = nn.Linear(128, 128, bias=False)
    orig_w = layer.weight.data.clone()
    quantizer = GPTQQuantizer(layer, bits=4, group_size=32)

    x = torch.randn(64, 128)
    quantizer.add_batch(x)
    w_q = quantizer.quantize(damp=0.01)

    assert w_q.shape == (128, 128)
    assert not torch.equal(layer.weight.data, orig_w)

    with torch.no_grad():
        out_orig = torch.matmul(x, orig_w.t())
        out_q = layer(x)
        rel_err = torch.norm(out_orig - out_q) / torch.norm(out_orig)
        assert rel_err < 0.6


def test_quantize_model_gptq():
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(8, 32) for _ in range(3)]
    quantize_model_gptq(model, calib, bits=4)
    out = model(torch.randn(2, 32))
    assert out.shape == (2, 16)
