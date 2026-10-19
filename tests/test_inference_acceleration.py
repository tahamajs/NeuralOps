#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Inference acceleration and auto-merge unit and hardware verification test.
Tests:
- merge_for_inference on direct_svd and PiSSALinear
- CUDAGraphDecodeRunner capture and step replay (with GPU auto-detection)
"""

import pytest
import torch
import torch.nn as nn

from neuralops.compression_engine.svd.direct_svd import merge_for_inference, DirectSVDCompressor
from neuralops.trainer.peft_recovery.svd_adapters import PiSSALinear
from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM, CUDAGraphDecodeRunner


def test_merge_for_inference_svd_linear():
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )
    compressor = DirectSVDCompressor(model, rank=16, fallback_to_original=False)
    comp_model, stats = compressor.compress()

    has_svd = any(child.__class__.__name__ in ("SVDLinear", "SVDLinearLayer") for _, child in comp_model.named_modules())
    assert has_svd

    x = torch.randn(2, 64)
    with torch.no_grad():
        out_before = comp_model(x)

    merged_model = merge_for_inference(comp_model, rank_threshold=None)

    has_svd_after = any(child.__class__.__name__ in ("SVDLinear", "SVDLinearLayer") for _, child in merged_model.named_modules())
    assert not has_svd_after

    with torch.no_grad():
        out_after = merged_model(x)

    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_merge_for_inference_pissa():
    lin = nn.Linear(32, 64)
    pissa = PiSSALinear(lin, rank=8)

    x = torch.randn(4, 32)
    with torch.no_grad():
        out_before = pissa(x)

    dummy_container = nn.Sequential(pissa)
    merged_container = merge_for_inference(dummy_container)

    assert isinstance(merged_container[0], nn.Linear)
    with torch.no_grad():
        out_after = merged_container(x)

    assert torch.allclose(out_before, out_after, atol=1e-4)


def test_cuda_graph_decode_runner():
    config = NeuralOpsConfig(
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        attention_types=["linear", "linear"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralOpsForCausalLM(config).to(device)
    model.eval()

    runner = CUDAGraphDecodeRunner(model, batch_size=1, device=device)
    runner.capture()

    if torch.cuda.is_available():
        assert runner.is_captured
    else:
        assert not runner.is_captured

    tok = torch.tensor([[10]], device=device)
    logits = runner.decode_step(tok)
    assert logits.shape == (1, 1, 500)
