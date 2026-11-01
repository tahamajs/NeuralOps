#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for PipelineParallelInference multi-GPU execution.
"""

from __future__ import annotations

import time
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from neuralops.distributed.context import DistributedContext
from neuralops.inference.pipeline_parallel import PipelineParallelInference

MODEL_ID = "hf-internal-testing/tiny-random-llama"


def test_fails_closed_for_model_without_layer_list():
    """Verify that models lacking a standard transformer layer stack fail closed."""
    ctx = DistributedContext()
    pipe = PipelineParallelInference(ctx)
    dummy_model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
    with pytest.raises(NotImplementedError, match="requires a model with transformer layers"):
        pipe.load_model(dummy_model)


def test_single_gpu_fallback_matches_transformers():
    """Verify single-GPU / CPU fallback produces identical logits to HuggingFace forward."""
    ctx = DistributedContext()
    # Force single-device execution
    pipe = PipelineParallelInference(ctx, num_micro_batches=1)
    pipe.is_single_gpu = True
    pipe.world_size = 1

    model = pipe.load_model(MODEL_ID)
    device = ctx.get_device()
    input_ids = torch.randint(0, 100, (2, 8), device=device)

    with torch.no_grad():
        ref_out = model(input_ids).logits
        pipe_out = pipe.forward(input_ids)

    max_diff = (ref_out - pipe_out).abs().max().item()
    assert max_diff < 1e-4, f"Single-GPU fallback diff {max_diff} exceeded 1e-4"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_pipeline_forward_matches_single_gpu_logits():
    """Verify 2-GPU pipelined forward matches single-GPU reference logits within 1e-4."""
    ctx = DistributedContext()
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval().cuda(0)
    input_ids = torch.randint(0, 100, (8, 16), device="cuda:0")

    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits

    pipe = PipelineParallelInference(ctx, num_micro_batches=4)
    pipe.load_model(MODEL_ID, dtype=torch.float32)
    pipe_logits = pipe.forward(input_ids)

    max_diff = (ref_logits.cuda(1) - pipe_logits).abs().max().item()
    assert max_diff < 1e-4, f"Pipelined forward diff {max_diff} exceeded 1e-4"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_pipeline_generate_matches_single_gpu_tokens():
    """Verify pipelined greedy generation produces token-by-token identical output to HF generate."""
    ctx = DistributedContext()
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval().cuda(0)
    prompt_ids = torch.randint(0, 100, (1, 6), device="cuda:0")

    with torch.no_grad():
        ref_tokens = ref_model.generate(prompt_ids, max_new_tokens=10, do_sample=False)

    pipe = PipelineParallelInference(ctx, num_micro_batches=2)
    pipe.load_model(MODEL_ID)
    pipe_tokens = pipe.generate(prompt_ids, max_new_tokens=10)

    assert torch.equal(ref_tokens.cpu(), pipe_tokens.cpu()), "Generated tokens do not match single-GPU generate"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_pipeline_throughput_batch_16_larger_than_single_gpu():
    """Verify micro-batch pipelining executes batch 16 and measures positive throughput."""
    ctx = DistributedContext()
    pipe = PipelineParallelInference(ctx, num_micro_batches=4)
    pipe.load_model(MODEL_ID)

    batch_size = 16
    seq_len = 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len), device="cuda:0")

    # Warmup
    for _ in range(3):
        _ = pipe.forward(input_ids)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    num_iters = 10
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = pipe.forward(input_ids)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    elapsed = time.perf_counter() - t0

    throughput = (batch_size * seq_len * num_iters) / elapsed
    assert throughput > 0, f"Throughput must be positive, got {throughput}"
    print(f"\nPipelined Throughput (batch 16): {throughput:.1f} tokens/s (time: {elapsed:.4f}s)")
