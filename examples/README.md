#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# NeuralOps examples suite (v2.0.3)

# NeuralOps Examples Suite

The NeuralOps examples suite provides **35 self-contained, reproducible, offline-friendly demonstration scripts** covering all subsystems of the framework: model surgery, low-rank compression, quantization, distillation, distributed training, high-throughput serving, kernel dispatch, compiler graphs, and runtime resilience.

All scripts are engineered to execute reliably on CPU (macOS/Linux) without external network calls or GPU requirements, while fully supporting GPU acceleration via `--device cuda` when available (with strict fail-closed verification).

---

## Suite Summary Table (35 Examples)

| Script | Subsystem | Demonstrates | Target Device | Execution Command |
|---|---|---|---|---|
| [`01_attention_surgery.py`](01_attention_surgery.py) | Architecture Surgery | Attention replacement across 5 families (cosformer, gla, gdn, fla, transformer) | CPU / MPS / CUDA | `python examples/01_attention_surgery.py` |
| [`02_model_compression.py`](02_model_compression.py) | Low-Rank SVD | SVD-LLM v2, Whitened SVD, HNC-SVD rank allocation, and parameter accounting | CPU / MPS / CUDA | `python examples/02_model_compression.py` |
| [`03_lora_finetuning.py`](03_lora_finetuning.py) | Parameter-Efficient Tuning | LoRA adapter injection into post-surgery attention and MLP layers | CPU / MPS / CUDA | `python examples/03_lora_finetuning.py` |
| [`04_knowledge_distillation.py`](04_knowledge_distillation.py) | Distillation Recovery | Teacher-student KL-divergence distillation and logits matching | CPU / MPS / CUDA | `python examples/04_knowledge_distillation.py` |
| [`05_fla_style_benchmarking.py`](05_fla_style_benchmarking.py) | Microbenchmarking | Cold-cache latency benchmarking across attention mechanisms | CPU / MPS / CUDA | `python examples/05_fla_style_benchmarking.py` |
| [`06_export_and_push.py`](06_export_and_push.py) | Model Serialization | SafeTensors, ONNX, and GGUF export with surgery manifest verification | CPU / MPS / CUDA | `python examples/06_export_and_push.py` |
| [`07_custom_attention.py`](07_custom_attention.py) | Extensibility Engine | @register_attention registration of custom attention mechanisms | CPU / MPS / CUDA | `python examples/07_custom_attention.py` |
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
| [`21_full_model_surgery.py`](21_full_model_surgery.py) | Full-Model Surgery | Replace attention in every model layer, then verify finite forward and backward gradients | CPU / MPS / CUDA | `python examples/21_full_model_surgery.py` |
| [`22_surgery_baseline_report.py`](22_surgery_baseline_report.py) | Surgery Integrity Report | Compare pre/post-surgery outputs and generate a machine-readable JSON report | CPU / MPS / CUDA | `python examples/22_surgery_baseline_report.py` |
| [`23_universal_surgery_director_ten_laws.py`](23_universal_surgery_director_ten_laws.py) | Universal Surgery Director | Automated compliance enforcement of the Ten Inviolable Laws of Attention Surgery | CPU / MPS / CUDA | `python examples/23_universal_surgery_director_ten_laws.py` |
| [`24_deepseek_mla_surgery_and_healing.py`](24_deepseek_mla_surgery_and_healing.py) | DeepSeek MLA Surgery | Dual-stream latent projection, decoupled RoPE preservation, and micro-healing | CPU / MPS / CUDA | `python examples/24_deepseek_mla_surgery_and_healing.py` |
| [`25_fx_graph_rewriter_compiler.py`](25_fx_graph_rewriter_compiler.py) | Torch FX Rewriter | Graph capture, symbolic subgraph pattern matching, and dead-code elimination | CPU / MPS / CUDA | `python examples/25_fx_graph_rewriter_compiler.py` |
| [`26_streaming_shadow_twin_distillation.py`](26_streaming_shadow_twin_distillation.py) | Zero-OOM Streaming Distillation | Shadow twin detached teacher caching and sequence micro-chunking | CPU / MPS / CUDA | `python examples/26_streaming_shadow_twin_distillation.py` |
| [`27_huggingface_polymorphic_cache_bridge.py`](27_huggingface_polymorphic_cache_bridge.py) | Polymorphic Generation Bridge | Seamless inter-framework cache bridge for HuggingFace model.generate() integration | CPU / MPS / CUDA | `python examples/27_huggingface_polymorphic_cache_bridge.py` |
| [`28_smart_trainer_automax_vram.py`](28_smart_trainer_automax_vram.py) | AutoMax VRAM Smart Trainer | Hardware headroom saturation probe and adaptive gradient accumulation | CPU / MPS / CUDA | `python examples/28_smart_trainer_automax_vram.py` |
| [`29_dora_healing_directional_adaptation.py`](29_dora_healing_directional_adaptation.py) | DoRA Directional Adaptation | Weight-decomposed low-rank adaptation for accelerated post-surgery recovery | CPU / MPS / CUDA | `python examples/29_dora_healing_directional_adaptation.py` |
| [`30_ttt_online_test_time_training.py`](30_ttt_online_test_time_training.py) | Online Test-Time Training (TTT) | Adaptive self-supervised sequence fine-tuning with instant transactional rollback | CPU / MPS / CUDA | `python examples/30_ttt_online_test_time_training.py` |
| [`31_fisher_damping_natural_gradient.py`](31_fisher_damping_natural_gradient.py) | Fisher Natural Gradient Damping | Empirical Fisher diagonal damping for ill-conditioned surgical layers | CPU / MPS / CUDA | `python examples/31_fisher_damping_natural_gradient.py` |
| [`32_simpo_orpo_preference_alignment.py`](32_simpo_orpo_preference_alignment.py) | Preference Alignment (SimPO/ORPO) | Reference-free policy alignment cutting 50% training VRAM overhead | CPU / MPS / CUDA | `python examples/32_simpo_orpo_preference_alignment.py` |
| [`33_triton_26_hardware_kernels.py`](33_triton_26_hardware_kernels.py) | Unified Triton 26 Kernels | Hardware profiling across 62 registered unified Triton operators with zero-fake fallbacks | CPU / MPS / CUDA | `python examples/33_triton_26_hardware_kernels.py` |
| [`34_circuit_breaker_runtime_resilience.py`](34_circuit_breaker_runtime_resilience.py) | Circuit Breaker Runtime FSM | Autonomous 3-state failure mitigation FSM with automatic 25ms cooldown recovery | CPU / MPS / CUDA | `python examples/34_circuit_breaker_runtime_resilience.py` |
| [`35_cuda_graph_decode_runner.py`](35_cuda_graph_decode_runner.py) | CUDA Graph Decode Runner | Static 1-token decode graph runner eliminating host launch overhead | CPU / MPS / CUDA | `python examples/35_cuda_graph_decode_runner.py` |

---

## Shared Infrastructure: `examples/_shared/`

To guarantee standard verification protocols and avoid code duplication across examples, all scripts utilize the `examples/_shared/` package:

- **`mock_models.py`**: Lightweight, deterministically initialized transformer architectures (`TinyTransformerConfig`, `TinyTransformerForCausalLM`, `TinyTransformerBlock`).
- **`offline_data.py`**: Synthetic deterministic text tokens and embeddings generated with fixed PRNG seeds (`make_synthetic_batch`, `make_synthetic_dataset`).
- **`verification.py`**: Rigorous finite checks, forward/backward gradient assertions, numerical tolerance bounds (`assert_finite`, `assert_close`, `check_subspace_orthogonality`).
- **`benchmark.py`**: High-precision cold-cache and warm-cache latency profiling with standard deviation error bars (`ColdCacheBenchmark`).
- **`gpu.py`**: Safe hardware discovery and fail-closed accelerator guards (`resolve_device`, `require_cuda`).
- **`report.py`**: Standardized console printing and structured JSON report persistence (`print_header`, `print_metrics_table`, `save_report_json`).

---

## Quickstart Run

Run any example script directly from the repository root:

```bash
# Architecture Surgery
python examples/01_attention_surgery.py

# Universal Surgery Director & Ten Laws
python examples/23_universal_surgery_director_ten_laws.py

# DeepSeek Multi-Head Latent Attention (MLA) Surgery
python examples/24_deepseek_mla_surgery_and_healing.py

# Advanced Optimizers (Muon & Stiefel)
python examples/14_advanced_optimizers_muon_stiefel.py

# Autonomous Circuit Breaker Runtime Resilience
python examples/34_circuit_breaker_runtime_resilience.py

# CUDA Graph Decode Engine
python examples/35_cuda_graph_decode_runner.py
```
