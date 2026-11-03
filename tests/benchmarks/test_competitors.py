#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Industry Competitor Benchmark Blueprints (Phases 61 to 80).

Validates:
1. Package exports in benchmarks.competitors.__all__
2. Registration of all 10 competitor benchmarks in neuralops.benchmarks.registry
3. FlashAttention standard peak TFLOPS calculations and bounds (Phase 62-63)
4. vLLM continuous Poisson serving simulation & PagedKVCache allocation (Phase 64-66)
5. SGLang RadixAttention prefix caching and trie lookup (Phase 67-68)
6. Megatron-LM MFU metrics & TP/SP forward-backward simulation (Phase 69-70)
7. DeepSpeed ZeRO-3 parameter unpartitioning & PCIe streaming (Phase 71-72)
8. TensorRT SQNR in dB & Frobenius norm distortion gates (Phase 73-74)
9. LM-Evaluation-Harness multi-choice log-likelihood evaluation (Phase 75-76)
10. Lighteval fast streaming tokenizer & reproducible manifest (Phase 77)
11. RULER variable-depth multi-needle passkey retrieval (Phase 78)
12. Medusa multi-head speculative tree decoding & TPS speedup (Phase 79)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    get_benchmark,
    list_benchmarks,
)
import benchmarks.competitors as competitors
from benchmarks.competitors import (
    benchmark_deepspeed_zero3_overhead,
    benchmark_flash_attn_tflops,
    benchmark_lighteval_fast_stream,
    benchmark_lm_eval_mmlu_gsm8k,
    benchmark_medusa_speculative_tree,
    benchmark_megatron_tp_sp,
    benchmark_pcie_offload_bandwidth,
    benchmark_radix_prefix_cache,
    benchmark_ruler_long_context,
    benchmark_tensorrt_sqnr,
    benchmark_vllm_poisson_serving,
    build_few_shot_prompt,
    compute_megatron_mfu,
    compute_tp_sp_communication_volume,
    evaluate_choice_loglikelihoods,
    evaluate_medusa_tree_acceptance,
    generate_dataset_reproducibility_manifest,
    generate_lighteval_manifest,
    generate_ruler_multi_needle_prompt,
    quantize_simulate,
    run_deepspeed_zero3_benchmark,
    run_lighteval_fast_stream_benchmark,
    run_lm_eval_benchmark,
    run_medusa_speculative_benchmark,
    run_megatron_tp_sp_benchmark,
    run_ruler_benchmark,
    run_tensorrt_quant_benchmark,
    run_tensorrt_quant_sqnr_suite,
)


def test_package_exports() -> None:
    """Verify that benchmarks.competitors exposes all 10 suites and their public symbols."""
    expected_callables = [
        "benchmark_flash_attn_tflops",
        "benchmark_vllm_poisson_serving",
        "run_vllm_serving_benchmark",
        "benchmark_radix_prefix_cache",
        "run_sglang_radix_benchmark",
        "benchmark_megatron_tp_sp",
        "compute_megatron_mfu",
        "compute_tp_sp_communication_volume",
        "run_megatron_tp_sp_benchmark",
        "benchmark_deepspeed_zero3_overhead",
        "benchmark_pcie_offload_bandwidth",
        "run_deepspeed_zero3_benchmark",
        "benchmark_tensorrt_sqnr",
        "quantize_simulate",
        "run_tensorrt_quant_benchmark",
        "run_tensorrt_quant_sqnr_suite",
        "benchmark_lm_eval_mmlu_gsm8k",
        "build_few_shot_prompt",
        "evaluate_choice_loglikelihoods",
        "run_lm_eval_benchmark",
        "benchmark_lighteval_fast_stream",
        "generate_dataset_reproducibility_manifest",
        "generate_lighteval_manifest",
        "run_lighteval_fast_stream_benchmark",
        "benchmark_ruler_long_context",
        "generate_ruler_multi_needle_prompt",
        "run_ruler_benchmark",
        "benchmark_medusa_speculative_tree",
        "evaluate_medusa_tree_acceptance",
        "run_medusa_speculative_benchmark",
    ]
    for sym in expected_callables:
        assert hasattr(competitors, sym), f"Missing symbol {sym} in benchmarks.competitors"
        assert sym in competitors.__all__, f"{sym} missing from __all__"


def test_benchmark_registry_bindings() -> None:
    """Verify all 10 competitor benchmark runners are registered in BENCHMARK_REGISTRY."""
    expected_runners = [
        "flash_attention_standard",
        "vllm_poisson_serving",
        "sglang_radix_cache",
        "megatron_tp_sp",
        "deepspeed_zero3_overhead",
        "tensorrt_quant_sqnr",
        "lm_eval_mmlu_gsm8k",
        "lighteval_fast_stream",
        "ruler_long_context",
        "medusa_speculative_tree",
    ]
    all_benchmarks = list_benchmarks()
    for name in expected_runners:
        assert name in BENCHMARK_REGISTRY, f"Runner {name} not found in BENCHMARK_REGISTRY"
        assert name in all_benchmarks, f"Runner {name} not listed in list_benchmarks()"
        fn = get_benchmark(name)
        assert callable(fn), f"Runner {name} must be callable"


def test_flash_attention_standard_cpu_fallback() -> None:
    """Test FlashAttention standard harness execution on CPU."""
    res = benchmark_flash_attn_tflops(
        batch=2,
        heads=4,
        seq_len=64,
        dim=32,
        warmup=2,
        iters=5,
        device="cpu",
        peak_tflops=10.0,
    )
    assert res["benchmark"] == "flash_attention_standard"
    assert "latency_ms" in res
    assert res["latency_ms"] > 0
    assert "fwd_latency_ms" in res
    assert res["fwd_latency_ms"] > 0
    assert "bwd_latency_ms" in res
    assert res["bwd_latency_ms"] >= 0
    assert "tflops" in res
    assert res["tflops"] >= 0
    assert "pct_peak" in res
    assert res["batch_size"] == 2
    assert res["num_heads"] == 4
    assert res["seq_len"] == 64
    assert res["head_dim"] == 32
    assert "fwd_flops" in res
    assert "bwd_flops" in res
    assert "total_flops" in res
    assert res["total_flops"] == res["fwd_flops"] + res["bwd_flops"]


def test_vllm_poisson_serving() -> None:
    """Test vLLM Poisson serving simulation, TTFT/TPOT percentiles, and cache lifecycle."""
    res = benchmark_vllm_poisson_serving(
        arrival_rate_lambda=50.0,
        num_requests=10,
        max_seq=128,
        min_seq=32,
        decode_steps=4,
        num_blocks=256,
        block_size=16,
        num_heads=4,
        head_dim=32,
        device="cpu",
        seed=123,
    )
    assert res["benchmark"] == "vllm_poisson_serving"
    assert res["num_requests"] == 10
    assert "ttft_p50_ms" in res
    assert "ttft_p95_ms" in res
    assert "ttft_p99_ms" in res
    assert "tpot_p50_ms" in res
    assert "tpot_p95_ms" in res
    assert "tpot_p99_ms" in res
    assert res["ttft_p99_ms"] >= res["ttft_p50_ms"]
    assert res["tpot_p99_ms"] >= res["tpot_p50_ms"]
    assert res["throughput_req_sec"] > 0
    assert res["throughput_tokens_sec"] > 0


def test_sglang_radix_cache() -> None:
    """Test SGLang RadixAttention prefix caching, lookup latency, and hit rate."""
    res = benchmark_radix_prefix_cache(
        common_prefix_len=64,
        query_suffix_len=16,
        num_queries=20,
        prefix_diversity=2,
        device="cpu",
        seed=42,
    )
    assert res["benchmark"] == "sglang_radix_cache"
    assert "prefix_hit_rate_pct" in res
    assert 0.0 <= res["prefix_hit_rate_pct"] <= 100.0
    assert "mean_lookup_us" in res
    assert res["mean_lookup_us"] >= 0
    assert "p95_lookup_us" in res
    assert res["p95_lookup_us"] >= 0
    assert res["mean_matched_tokens"] >= 0


def test_megatron_tp_sp_and_mfu() -> None:
    """Test Megatron-LM MFU calculation formula and distributed module simulation."""
    # Test formula
    mfu_calc = compute_megatron_mfu(
        params_count=1_000_000,
        tokens_per_sec=10_000.0,
        peak_gpu_tflops=100.0,
        num_gpus=1,
    )
    # 6 * 1e6 * 1e4 = 6e10 FLOPs. peak = 100 * 1e12 = 1e14 FLOPs.
    # ratio = 6e10 / 1e14 = 6e-4 = 0.06%
    assert abs(mfu_calc["mfu_pct"] - 0.06) < 1e-4

    # Test communication volume formula
    # B=2, S=64, H=128, TP=2, bytes=2 (fp16)
    # factor = 2 * (2 - 1) / 2 = 1.0
    # tokens = 2 * 64 = 128
    # allreduce_bytes = 1.0 * 128 * 128 * 2 = 32768 bytes
    vol = compute_tp_sp_communication_volume(
        batch_size=2,
        seq_len=64,
        hidden_size=128,
        tp_size=2,
        dtype_bytes=2,
    )
    assert vol["allreduce_volume_bytes"] == 32768.0
    assert vol["layer_fwd_volume_mb"] > 0
    assert vol["factor"] == 1.0

    # Test runner execution
    res = benchmark_megatron_tp_sp(
        hidden_size=64,
        intermediate_size=128,
        seq_len=32,
        batch_size=2,
        tp_size=1,
        iters=3,
        warmup=1,
        peak_gpu_tflops=50.0,
        device="cpu",
    )
    assert res["benchmark"] == "megatron_tp_sp"
    assert res["tokens_per_sec"] > 0
    assert "mfu_pct" in res
    assert res["mfu_pct"] >= 0
    assert res["latency_ms"] > 0
    assert "allreduce_volume_mb" in res

    # Verify alias callable
    res_alias = run_megatron_tp_sp_benchmark(
        hidden_size=64,
        intermediate_size=128,
        seq_len=16,
        batch_size=1,
        tp_size=1,
        iters=1,
        warmup=1,
        device="cpu",
    )
    assert res_alias["benchmark"] == "megatron_tp_sp"


def test_deepspeed_zero3_overhead() -> None:
    """Test DeepSpeed ZeRO-3 parameter unpartitioning and gather bandwidth."""
    res = benchmark_deepspeed_zero3_overhead(
        num_shards=2,
        shard_size_mb=4,
        iters=3,
        warmup=1,
        device="cpu",
    )
    assert res["benchmark"] == "deepspeed_zero3_overhead"
    assert res["num_shards"] == 2
    assert res["total_param_mb"] == 8
    assert res["gather_latency_ms"] > 0
    assert res["effective_bandwidth_gb_sec"] > 0
    assert res["memory_footprint_reduction_ratio"] == 0.5

    # Verify alias callable
    res_alias = run_deepspeed_zero3_benchmark(
        num_shards=4,
        shard_size_mb=2,
        iters=1,
        warmup=1,
        device="cpu",
    )
    assert res_alias["benchmark"] == "deepspeed_zero3_overhead"
    assert res_alias["memory_footprint_reduction_ratio"] == 0.25


def test_tensorrt_quant_sqnr() -> None:
    """Test TensorRT SQNR in dB calculation and Frobenius distortion gating."""
    orig = torch.randn(100, 100)
    # Very minor noise -> High SQNR
    quant = orig + torch.randn(100, 100) * 0.001
    stats = benchmark_tensorrt_sqnr(orig, quant, min_sqnr_threshold_db=30.0, max_frobenius_threshold=0.05)
    assert stats["sqnr_db"] > 40.0
    assert stats["rel_frobenius"] < 0.01
    assert stats["passed"] is True

    # Heavy noise -> Low SQNR
    quant_bad = orig + torch.randn(100, 100) * 1.0
    stats_bad = benchmark_tensorrt_sqnr(orig, quant_bad, min_sqnr_threshold_db=30.0, max_frobenius_threshold=0.05)
    assert stats_bad["sqnr_db"] < 20.0
    assert stats_bad["passed"] is False

    # Full suite execution
    suite_res = run_tensorrt_quant_sqnr_suite(
        layer_shapes=[(64, 64)],
        formats=["fp8_e4m3", "fp8_e5m2", "int8", "int4", "int4_mx"],
        device="cpu",
        seed=42,
    )
    assert suite_res["benchmark"] == "tensorrt_quant_sqnr"
    assert suite_res["evaluated_layers_count"] == 1
    assert "all_standard_gates_passed" in suite_res

    # Test individual formats via quantize_simulate
    t = torch.randn(32, 32)
    for fmt in ["fp8_e4m3", "fp8_e5m2", "int8", "int4", "int4_mx"]:
        q = quantize_simulate(t, format_type=fmt)
        res = benchmark_tensorrt_sqnr(t, q)
        assert res["sqnr_db"] > 0.0
        assert res["rel_frobenius"] >= 0.0



def test_lm_eval_mmlu_gsm8k() -> None:
    """Test LM-Eval log-likelihood choice selection and benchmark suite."""
    # Synthetic model
    class SimpleLM(nn.Module):
        def __init__(self, vocab: int = 100, dim: int = 32):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.fc = nn.Linear(dim, vocab)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.emb(x))

    model = SimpleLM()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    cand_a = torch.tensor([[4, 5]], dtype=torch.long)
    cand_b = torch.tensor([[6, 7]], dtype=torch.long)

    chosen_idx = evaluate_choice_loglikelihoods(
        model, prompt, [cand_a, cand_b], device="cpu", length_normalization=True
    )
    assert chosen_idx in (0, 1)

    chosen_idx_raw, scores = evaluate_choice_loglikelihoods(
        model, prompt, [cand_a, cand_b], device="cpu", length_normalization=False, return_scores=True
    )
    assert chosen_idx_raw in (0, 1)
    assert len(scores) == 2

    few_shot = build_few_shot_prompt([{"question": "Q1", "answer": "A1"}], "Q2")
    assert "Question: Q1" in few_shot
    assert "Question: Q2" in few_shot

    # Full benchmark runner with few-shot and length normalization
    res = benchmark_lm_eval_mmlu_gsm8k(
        model=None,
        num_samples=5,
        vocab_size=1000,
        hidden_size=64,
        num_choices=3,
        num_shots=1,
        length_normalization=True,
        device="cpu",
        seed=42,
    )
    assert res["benchmark"] == "lm_eval_mmlu_gsm8k"
    assert res["num_samples"] == 5
    assert res["num_shots"] == 1
    assert res["length_normalization"] is True
    assert 0.0 <= res["accuracy_pct"] <= 100.0


def test_lighteval_fast_stream() -> None:
    """Test Lighteval fast streaming throughput and manifest generation."""
    manifest = generate_lighteval_manifest(
        task_name="test_task",
        model_name="test_model",
        score=85.5,
        metrics={"metric_a": 123.4},
    )
    assert manifest["task"] == "test_task"
    assert manifest["model"] == "test_model"
    assert manifest["primary_score"] == 85.5
    assert len(manifest["manifest_hash"]) == 64

    data_manifest = generate_dataset_reproducibility_manifest(
        samples=["item 1", "item 2", "item 3"],
        dataset_name="test_dataset",
    )
    assert data_manifest["dataset_name"] == "test_dataset"
    assert data_manifest["sample_count"] == 3
    assert len(data_manifest["dataset_sha256"]) == 64

    res = benchmark_lighteval_fast_stream(
        num_samples=10,
        seq_len=32,
        batch_size=2,
        device="cpu",
        seed=42,
    )
    assert res["benchmark"] == "lighteval_fast_stream"
    assert res["num_samples"] == 10
    assert res["samples_per_sec"] > 0
    assert res["tokens_per_sec"] > 0
    assert "manifest" in res
    assert "dataset_manifest" in res
    assert len(res["dataset_manifest"]["dataset_sha256"]) == 64


def test_ruler_long_context() -> None:
    """Test RULER multi-needle prompt synthesis and long context evaluation."""
    # Test standard prompt synthesis
    text, needles = generate_ruler_multi_needle_prompt(context_length=500, num_needles=3)
    assert len(needles) == 3
    for n in needles:
        assert n["key"] in text
        assert n["value"] in text

    # Test explicit depth percentiles (10%, 25%, 50%, 75%, 90%)
    depth_pcts = [10.0, 25.0, 50.0, 75.0, 90.0]
    text_pct, needles_pct = generate_ruler_multi_needle_prompt(
        context_length=1000,
        depth_percentiles=depth_pcts,
    )
    assert len(needles_pct) == 5
    for i, n in enumerate(needles_pct):
        assert n["depth_pct"] == depth_pcts[i]
        assert n["key"] in text_pct
        assert n["value"] in text_pct

    # Run benchmark across multi-needle depths and context lengths up to 32k/128k emulation
    res = run_ruler_benchmark(
        context_lengths=[512, 1024, 2048],
        depth_percentiles=depth_pcts,
        device="cpu",
        seed=42,
    )
    assert res["benchmark"] == "ruler_long_context"
    assert res["total_needles_evaluated"] == 15
    assert res["overall_retrieval_accuracy_pct"] == 100.0
    assert len(res["length_profiles"]) == 3
    assert "accuracy_matrix" in res
    assert "512" in res["accuracy_matrix"]
    assert "10.0%" in res["accuracy_matrix"]["512"]
    assert res["accuracy_matrix"]["512"]["10.0%"] == 100.0


def test_medusa_speculative_tree() -> None:
    """Test Medusa speculative tree acceptance logic and benchmark runner."""
    # Test acceptance helper
    # 3 candidates: [10, 20, 30]
    # Logits: depth 0 argmax=10, depth 1 argmax=20, depth 2 argmax=99
    # Accepted tokens should be 2 + 1 bonus = 3
    candidates = [10, 20, 30]
    logits = torch.zeros(3, 100)
    logits[0, 10] = 10.0
    logits[1, 20] = 10.0
    logits[2, 99] = 10.0
    stats = evaluate_medusa_tree_acceptance(candidates, logits)
    assert stats["accepted_tokens"] == 3
    assert stats["perfect_match"] is False

    # Full benchmark runner
    res = run_medusa_speculative_benchmark(
        num_heads=3,
        tree_depth=3,
        num_steps=10,
        vocab_size=1000,
        acceptance_bias=0.8,
        device="cpu",
        seed=42,
    )
    assert res["benchmark"] == "medusa_speculative_tree"
    assert res["total_tokens_emitted"] >= 10
    assert res["speculative_speedup_factor"] >= 1.0
    assert res["mean_accepted_tokens_per_step"] >= 1.0
