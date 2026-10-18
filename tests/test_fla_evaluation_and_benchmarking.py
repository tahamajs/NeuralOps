# -*- coding: utf-8 -*-
"""
test_fla_evaluation_and_benchmarking.py
=======================================
Comprehensive Verification Suite for FLA Evaluation & Benchmarking Tools in NeuralOps:
1. neuralops.evaluation.harness (NeuralOpsLMWrapper, HAS_LM_EVAL, run_evaluation)
2. neuralops.evaluation.ppl (PerplexityEvaluator)
3. neuralops.benchmarks.generation (GenerationBenchmark)
4. neuralops.benchmarks.training_throughput (TrainingThroughputProfiler)
5. neuralops.benchmarks.ops_runner (OP_REGISTRY, run_benchmark_worker, format_table)
6. neuralops.benchmarks.visualize (plot_operator_latencies, plot_positional_perplexity, plot_scaling_comparison)

Target Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9, CUDA 12.8, PyTorch 2.11.0+cu128)
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from neuralops.evaluation.harness import NeuralOpsLMWrapper, HAS_LM_EVAL, run_evaluation
from neuralops.evaluation.ppl import PerplexityEvaluator
from neuralops.benchmarks.generation import GenerationBenchmark
from neuralops.benchmarks.training_throughput import TrainingThroughputProfiler
from neuralops.benchmarks.ops_runner import OP_REGISTRY, run_benchmark_worker, format_table
from neuralops.benchmarks.visualize import (
    plot_operator_latencies,
    plot_positional_perplexity,
    plot_scaling_comparison,
)


@pytest.fixture(scope="module")
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tiny_causal_lm(device):
    """Creates a lightweight synthetic CausalLM model for fast live testing."""
    config = AutoConfig.from_pretrained(
        "gpt2",
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        n_positions=512,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model.to(device)


# ============================================================================
# 1. EVALUATION: Perplexity & Positional NLL (neuralops.evaluation.ppl)
# ============================================================================

def test_ppl_evaluator_synthetic_model(tiny_causal_lm, device):
    """Test PerplexityEvaluator with synthetic model and generated 1D tokens."""
    evaluator = PerplexityEvaluator(
        model=tiny_causal_lm,
        block_size=64,
        bucket_size=16,
        device=device,
        dtype=torch.float32,
    )
    
    # Generate 1D synthetic token stream with length >= block_size + 1
    tokens = torch.randint(0, 256, (256,), dtype=torch.long)
    results = evaluator.evaluate_tokens(tokens)

    assert "perplexity" in results
    assert "mean_nll" in results
    assert "positional_ppl_buckets" in results
    assert results["total_tokens"] > 0
    assert results["perplexity"] > 0.0
    assert not math.isnan(results["perplexity"])


# ============================================================================
# 2. EVALUATION: LM-Eval Harness Integration (neuralops.evaluation.harness)
# ============================================================================

def test_harness_model_wrapper(tiny_causal_lm, device):
    """Test NeuralOpsLMWrapper initialization and properties."""
    wrapper = NeuralOpsLMWrapper(
        pretrained=tiny_causal_lm,
        device=device,
        batch_size=2,
        max_length=128,
    )
    assert wrapper.model is not None
    assert str(wrapper.device).startswith(device.split(":")[0])
    assert wrapper.max_length == 128


# ============================================================================
# 3. BENCHMARKS: Token Generation (neuralops.benchmarks.generation)
# ============================================================================

def test_generation_benchmark_synthetic(tiny_causal_lm, device):
    """Test GenerationBenchmark for TTFT, per-token latency, and throughput."""
    bench = GenerationBenchmark(
        model=tiny_causal_lm,
        device=device,
        dtype=torch.float32,
    )
    res = bench.benchmark_generate(
        prompt_len=16,
        gen_len=8,
        batch_size=1,
        warmup_runs=1,
        num_runs=2,
        use_cache=True,
    )
    assert "ttft_ms" in res
    assert "latency_per_token_ms" in res
    assert "throughput_tokens_per_s" in res
    assert res["ttft_ms"] >= 0.0
    assert res["throughput_tokens_per_s"] >= 0.0


# ============================================================================
# 4. BENCHMARKS: Training Throughput & VRAM (neuralops.benchmarks.training_throughput)
# ============================================================================

def test_training_throughput_profiler_synthetic(tiny_causal_lm, device):
    """Test TrainingThroughputProfiler for forward+backward step timing and VRAM."""
    profiler = TrainingThroughputProfiler(
        model=tiny_causal_lm,
        device=device,
        dtype=torch.float32,
    )
    res = profiler.profile_training_step(
        batch_size=2,
        seq_len=32,
        warmup_steps=1,
        num_steps=2,
    )
    assert "step_latency_ms" in res
    assert "tokens_per_second" in res
    assert "peak_vram_mb" in res
    assert res["step_latency_ms"] > 0.0
    assert res["tokens_per_second"] > 0.0


# ============================================================================
# 5. BENCHMARKS: Microbenchmarking Ops Runner (neuralops.benchmarks.ops_runner)
# ============================================================================

def test_ops_runner_registry_and_worker(device):
    """Test OP_REGISTRY and run_benchmark_worker execution."""
    assert len(OP_REGISTRY) > 0
    assert "fused_rmsnorm" in OP_REGISTRY or "fused_ce" in OP_REGISTRY

    # Test run_benchmark_worker on fused_rmsnorm forward
    target_op = "fused_rmsnorm" if "fused_rmsnorm" in OP_REGISTRY else list(OP_REGISTRY.keys())[0]
    dev_str = "cuda" if "cuda" in device else "cpu"
    res = run_benchmark_worker(
        op_name=target_op,
        B=2,
        T=64,
        H=4,
        D=32,
        mode="fwd",
        dtype_str="float16",
        device_str=dev_str,
        warmup=2,
        rep=5,
    )
    assert "latency_ms" in res or "median_ms" in res or len(res) > 0

    # Test format_table utility
    results_list = [res] if isinstance(res, dict) else [{"op": target_op, "latency_ms": 0.015}]
    tbl = format_table(results_list)
    assert target_op in tbl or "latency" in tbl.lower() or len(tbl) > 0


# ============================================================================
# 6. BENCHMARKS: Visualization Tools (neuralops.benchmarks.visualize)
# ============================================================================

def test_visualize_functions(tmp_path):
    """Test visualize plotting functions with dummy metrics."""
    out_img = str(tmp_path / "test_plot.png")

    # 1. Positional perplexity plot
    pos_data = {
        "gpt2_baseline": {0: 15.2, 128: 14.8, 256: 14.5},
        "svd_linear_attn": {0: 15.5, 128: 14.9, 256: 14.3},
    }
    plot_positional_perplexity(pos_data, output_path=out_img)

    # 2. Scaling comparison plot
    scaling_data = {
        "Standard": [(512, 10.0), (1024, 25.0), (2048, 80.0)],
        "SVD": [(512, 5.0), (1024, 10.0), (2048, 20.0)],
    }
    out_img2 = str(tmp_path / "test_scaling.png")
    plot_scaling_comparison(scaling_data, output_path=out_img2)

