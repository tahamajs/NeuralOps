# NeuralOps Strategic Roadmap

This document outlines the architectural roadmap, development milestones, and release strategy for **NeuralOps** (`SVD_linear_Attention`).

---

## Current Release: v1.4.2 (October 2026) — Complete Consolidation

The v1.4.2 release completes the fundamental consolidation of all subquadratic attention kernels, model compression algorithms, serving runtimes, and training engines:

- **131 Attention Mechanisms** verified across 8 taxonomic families.
- **72 Model Compression Engines** spanning SVD, pruning, quantization, and KV-cache compression.
- **26 Triton Kernel Modules** (housing 62 unified Triton JIT operators across 4 tiers) with universal CPU/MPS fallbacks.
- **114 Native CUDA C++ Extensions** targeting Ada Lovelace (SM 8.9), Hopper (SM 90), and Blackwell (SM 100).
- **Flame-Style Multi-Dataset Streaming** (`--training.dataset`) & **Varlen Packing** (`cu_seqlens`).
- **Warmup-Stable-Decay (WSD)** Learning Rate Scheduler.
- **19-Function Canonical Top-Level API** (`neuralops`).
- **Full `torch.compile` Compatibility** with custom op Meta/FakeTensor registrations.
- **Serving Adapters** for vLLM and TGI runtimes (`VLLMAdapter`, `TGIAdapter`, `PagedKVCache`).
- **Zero-Shot Attention Injection** across 8 transformer architectures (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Mistral, Qwen2, T5).
- **Advanced Manifold Optimizers**: `Muon` (Newton-Schulz orthogonalization), `Lion`, `RiemannianStiefelOptimizer`, and `RiemannianAdam`.
- **Parametric Numerical Parity**: Validated cross-platform CPU/MPS/CUDA execution with zero skipped tests.

| Milestone Metric | v1.3.0 Baseline | v1.4.2 Delivered | Status |
| :--- | :--- | :--- | :--- |
| Canonical Attentions | 131 | 131 | **Verified 100%** |
| Compression Engines | 72 | 72 | **Verified 100%** |
| Triton Kernel Modules | 26 (62 Ops) | 26 (62 Ops) | **Verified 100%** |
| Native CUDA Kernels | 114 | 114 | **Verified 100%** |
| Executed Notebooks | 27 | 27 | **Verified 100%** |
| Dead Code LOC Pruned | 0 | 4,200+ LOC | **Pruned Cleanly** |
| Module Shadow Collisions | 3 collisions | 0 collisions | **Fully Resolved** |
| Test Parity Coverage | Partial (Mac Skipped) | 100% Parametric | **0 Failed, 0 Skipped** |

---

## v1.5.0 (Q1 2027) — Long-Context Scaling & Parallel Attention

### Strategic Focus
Scaling linear and recurrent attention mechanisms beyond 128k context lengths while preserving high distillation fidelity from pretrained dense models.

### Objectives
1. **Chunkwise Parallel Linear Attention**: Scale causal recurrent kernels across multiple GPUs using Megatron-style context parallelism for sequences exceeding 128k tokens.
2. **Selective State Scan v2**: Multi-head associative scan optimizations for Mamba-2 and Griffin architectures.
3. **Full GLUE & LRA Publication Sweeps**: Comprehensive empirical Pareto frontiers across all 131 attention mechanisms.
4. **Automated Hyperparameter Tuner**: Bayesian search integration with Ray Tune for layerwise hybrid allocation.

---

## v2.0.0 (Q4 2027) — Hardware & Edge Expansion

### Strategic Focus
Expanding beyond NVIDIA CUDA to native Apple Metal, Intel XPU, and quantized edge runtimes.
