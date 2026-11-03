#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# NeuralOps examples suite (v2.0.1)

# NeuralOps Examples Suite

The NeuralOps examples suite provides 22 self-contained, reproducible, offline-friendly demonstration scripts covering all subsystems of the framework: model surgery, low-rank compression, quantization, distillation, distributed training, high-throughput serving, kernel dispatch, and evaluation.

Examples 01–20 are designed to execute reliably on CPU (macOS/Linux) without external network calls or GPU requirements. Examples 21–22 add explicit full-model surgery and integrity-report workflows; when `--device cuda` is requested, they fail closed if CUDA is unavailable.

---

## Suite Summary Table (22 Examples)

| Script | Subsystem | Demonstrates | Target Device | Execution Command |
|---|---|---|---|---|
| [`01_attention_surgery.py`](01_attention_surgery.py) | Architecture Surgery | Attention replacement across 5 families (cosformer, gla, gdn, fla, transformer) | CPU / MPS / CUDA | `python examples/01_attention_surgery.py` |
| [`02_model_compression.py`](02_model_compression.py) | Low-Rank SVD | SVD-LLM v2, Whitened SVD, HNC-SVD rank allocation, and parameter accounting | CPU / MPS / CUDA | `python examples/02_model_compression.py` |
| [`03_lora_finetuning.py`](03_lora_finetuning.py) | Parameter-Efficient Tuning | LoRA adapter injection into post-surgery attention and MLP layers | CPU / MPS / CUDA | `python examples/03_lora_finetuning.py` |
| [`04_knowledge_distillation.py`](04_knowledge_distillation.py) | Distillation Recovery | Teacher-student KL-divergence distillation and logits matching | CPU / MPS / CUDA | `python examples/04_knowledge_distillation.py` |
| [`05_fla_style_benchmarking.py`](05_fla_style_benchmarking.py) | Microbenchmarking | Cold-cache latency benchmarking across attention mechanisms | CPU / MPS / CUDA | `python examples/05_fla_style_benchmarking.py` |
| [`06_export_and_push.py`](06_export_and_push.py) | Model Serialization | SafeTensors, ONNX, and GGUF export with surgery manifest verification | CPU / MPS / CUDA | `python examples/06_export_and_push.py` |
| [`07_custom_attention.py`](07_custom_attention.py) | Extensibility Engine | `@register_attention` registration of custom attention mechanisms | CPU / MPS / CUDA | `python examples/07_custom_attention.py` |
| [`08_grpo_reasoning.py`](08_grpo_reasoning.py) | Post-Training RL | GRPO loss computation and group relative policy optimization | CPU / MPS / CUDA | `python examples/08_grpo_reasoning.py` |
| [`09_multi_stage_model_surgery.py`](09_multi_stage_model_surgery.py) | Fluent Surgery DSL | Multi-stage pipeline: attention surgery -> SVD compression -> MLP surgery | CPU / MPS / CUDA | `python examples/09_multi_stage_model_surgery.py` |
| [`10_advanced_compression_zoo.py`](10_advanced_compression_zoo.py) | SVD Compression Zoo | Comparative analysis of 4 SVD algorithms (Standard, Whitened, Fisher, HNC) | CPU / MPS / CUDA | `python examples/10_advanced_compression_zoo.py` |
| [`11_quantization_fp8_int4_mx.py`](11_quantization_fp8_int4_mx.py) | Advanced Quantization | FP8 (E4M3/E5M2), INT4 uniform, and Microscaling (MX) simulation | CPU / MPS / CUDA | `python examples/11_quantization_fp8_int4_mx.py` |
| [`12_structured_pruning_and_token_shedding.py`](12_structured_pruning_and_token_shedding.py) | Sparsity & Pruning | Structured channel pruning, 2:4 sparsity, and dynamic token shedding | CPU / MPS / CUDA | `python examples/12_structured_pruning_and_token_shedding.py` |
| [`13_six_pillar_knowledge_distillation.py`](13_six_pillar_knowledge_distillation.py) | Multi-Pillar Distillation | 6-pillar distillation: Logits, Hidden, Attention, Spectral, Embedding, Task | CPU / MPS / CUDA | `python examples/13_six_pillar_knowledge_distillation.py` |
| [`14_advanced_optimizers_muon_stiefel.py`](14_advanced_optimizers_muon_stiefel.py) | Non-Euclidean Optimizers | Newton-Schulz matrix optimizer (Muon) and Stiefel manifold Riemannian Adam | CPU / MPS / CUDA | `python examples/14_advanced_optimizers_muon_stiefel.py` |
| [`15_serving_vllm_sglang_trtllm.py`](15_serving_vllm_sglang_trtllm.py) | Production Serving | vLLM PagedAttention, SGLang RadixAttention, and TRT-LLM plugin contract | CPU / MPS / CUDA | `python examples/15_serving_vllm_sglang_trtllm.py` |
| [`16_distributed_parallelism.py`](16_distributed_parallelism.py) | Distributed Training | Tensor Parallelism (TP column/row) and Context Parallelism ring-exchange | CPU / MPS / CUDA | `python examples/16_distributed_parallelism.py` |
| [`17_recipe_dsl_and_snapshot_rollback.py`](17_recipe_dsl_and_snapshot_rollback.py) | Surgery Recipes & Rollback | YAML recipe execution, state snapshot checkpointing, and instant rollback | CPU / MPS / CUDA | `python examples/17_recipe_dsl_and_snapshot_rollback.py` |
| [`18_three_level_kernel_dispatch.py`](18_three_level_kernel_dispatch.py) | Kernel Dispatch Architecture | 3-level fallback hierarchy: CUDA C++ -> Triton JIT -> Vectorized PyTorch | CPU / MPS / CUDA | `python examples/18_three_level_kernel_dispatch.py` |
| [`19_spectral_and_hessian_diagnostics.py`](19_spectral_and_hessian_diagnostics.py) | Landscape Diagnostics | Lanczos Hessian eigenspectrum estimation, top eigenvalue, and stable rank | CPU / MPS / CUDA | `python examples/19_spectral_and_hessian_diagnostics.py` |
| [`20_long_context_niah_eval.py`](20_long_context_niah_eval.py) | Long-Context Evaluation | Needle-in-a-Haystack (NIAH) retrieval evaluation across context lengths | CPU / MPS / CUDA | `python examples/20_long_context_niah_eval.py` |
| [`21_full_model_surgery.py`](21_full_model_surgery.py) | Full-Model Surgery | Replace attention in every model layer, then verify finite forward output and finite backward gradients | CPU / MPS / CUDA | `python examples/21_full_model_surgery.py --device cuda` |
| [`22_surgery_baseline_report.py`](22_surgery_baseline_report.py) | Surgery Integrity Report | Compare pre/post-surgery outputs and write a machine-readable JSON report | CUDA required by default | `python examples/22_surgery_baseline_report.py` |

---

## Shared Infrastructure: `examples/_shared/`

To guarantee standard verification protocols and avoid code duplication across examples, all scripts utilize the `examples/_shared/` package:

- **`mock_models.py`**: Provides `TinyClassifier` (a fast, deterministic transformer architecture with multi-head attention and MLPs), `tiny_inputs()` (synthetic token sequences), and reference logits.
- **`offline_data.py`**: Deterministic synthetic token generators, calibration dataset loaders, and streaming batches without internet access or Hugging Face downloads.
- **`benchmark.py`**: Robust CPU/GPU benchmarking functions, including `measure_latency()` with warmups, and `measure_vram_if_available()`.
- **`assertions.py`**: Parity gates and assertions (`assert_parity`, `assert_shape`, `assert_finite`) enforcing tight numerical tolerances (`atol`, `rtol`).
- **`report.py`**: Clean, formatted console reporting utilities (`section`, `table`, `kv`, `pass_fail`).
- **`device.py`**: Accelerator preference resolver (`get_backend_preference()`, `has_cuda()`, `has_mps()`).

## Authoring New Examples

When contributing new examples, always follow the canonical template provided in:
👉 [`examples/_template.py`](_template.py)

Key authoring rules:
1. Always inherit standard header license blocks referencing contributors.
2. Ensure strict offline CPU execution fallback so the script runs cleanly on any developer machine.
3. Use `examples._shared` helpers for models, metrics, and reporting.
4. If an optional third-party library or accelerator is absent, skip gracefully with `skip_if_missing()` and exit 0.
