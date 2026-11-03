#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Industry Competitor Benchmark Blueprints Package.

Exposes production-grade benchmark harnesses for 10 top industry competitor suites:
1. FlashAttention (Peak TFLOPS & Forward/Backward Saturation)
2. vLLM (Continuous Poisson Request Serving & PagedAttention)
3. SGLang (RadixAttention & Dynamic Prefix Cache Reuse)
4. Megatron-LM (Tensor & Sequence Parallel MFU Scaling)
5. DeepSpeed (ZeRO-3 Memory Partitioning & PCIe Streaming Overhead)
6. TensorRT-LLM (Microscaling & Signal-to-Quantization-Noise Ratio)
7. LM-Evaluation-Harness (EleutherAI Task Parity: MMLU / GSM8k)
8. HuggingFace Lighteval (Nanotron Fast Tokenizer Streaming & Manifest)
9. RULER (Long-Context Needle-In-A-Haystack 32k/128k Passkey Tracing)
10. Medusa (Self-Speculative Multi-Head Tree Decoding)
"""

from __future__ import annotations

from benchmarks.competitors.bench_deepspeed_zero3_overhead import (
    benchmark_deepspeed_zero3_overhead,
    benchmark_pcie_offload_bandwidth,
)
from benchmarks.competitors.bench_flash_attention_standard import (
    benchmark_flash_attn_tflops,
)
from benchmarks.competitors.bench_lighteval_fast_stream import (
    benchmark_lighteval_fast_stream,
    generate_dataset_reproducibility_manifest,
    generate_lighteval_manifest,
    run_lighteval_fast_stream_benchmark,
)
from benchmarks.competitors.bench_lm_eval_mmlu_gsm8k import (
    benchmark_lm_eval_mmlu_gsm8k,
    build_few_shot_prompt,
    evaluate_choice_loglikelihoods,
    run_lm_eval_benchmark,
)
from benchmarks.competitors.bench_medusa_speculative_tree import (
    benchmark_medusa_speculative_tree,
    evaluate_medusa_tree_acceptance,
)
from benchmarks.competitors.bench_megatron_tp_sp import (
    benchmark_megatron_tp_sp,
    compute_megatron_mfu,
)
from benchmarks.competitors.bench_ruler_long_context import (
    benchmark_ruler_long_context,
    generate_ruler_multi_needle_prompt,
)
from benchmarks.competitors.bench_sglang_radix_cache import (
    benchmark_radix_prefix_cache,
    run_sglang_radix_benchmark,
)
from benchmarks.competitors.bench_tensorrt_quant_sqnr import (
    benchmark_tensorrt_sqnr,
    quantize_simulate,
    run_tensorrt_quant_benchmark,
    run_tensorrt_quant_sqnr_suite,
)
from benchmarks.competitors.bench_vllm_poisson_serving import (
    benchmark_vllm_poisson_serving,
    run_vllm_serving_benchmark,
)

__all__ = [
    # FlashAttention
    "benchmark_flash_attn_tflops",
    # vLLM
    "benchmark_vllm_poisson_serving",
    "run_vllm_serving_benchmark",
    # SGLang
    "benchmark_radix_prefix_cache",
    "run_sglang_radix_benchmark",
    # Megatron-LM
    "benchmark_megatron_tp_sp",
    "compute_megatron_mfu",
    # DeepSpeed
    "benchmark_deepspeed_zero3_overhead",
    "benchmark_pcie_offload_bandwidth",
    # TensorRT-LLM
    "benchmark_tensorrt_sqnr",
    "quantize_simulate",
    "run_tensorrt_quant_benchmark",
    "run_tensorrt_quant_sqnr_suite",
    # LM-Evaluation-Harness
    "benchmark_lm_eval_mmlu_gsm8k",
    "build_few_shot_prompt",
    "evaluate_choice_loglikelihoods",
    "run_lm_eval_benchmark",
    # HuggingFace Lighteval
    "benchmark_lighteval_fast_stream",
    "generate_dataset_reproducibility_manifest",
    "generate_lighteval_manifest",
    "run_lighteval_fast_stream_benchmark",
    # RULER
    "benchmark_ruler_long_context",
    "generate_ruler_multi_needle_prompt",
    # Medusa Speculative Tree
    "benchmark_medusa_speculative_tree",
    "evaluate_medusa_tree_acceptance",
]
