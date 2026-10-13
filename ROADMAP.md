# NeuralOps Strategic Roadmap

This document outlines the architectural roadmap, development milestones, and release strategy for **NeuralOps** (`SVD_linear_Attention`).

---

## Current Release: v1.3.0 (October 2026) — Complete Consolidation

The v1.3.0 release completes the fundamental consolidation of all subquadratic attention kernels, model compression algorithms, serving runtimes, and training engines:

- **131 Attention Mechanisms** verified across 8 taxonomic families.
- **72 Model Compression Engines** spanning SVD, pruning, quantization, and KV-cache compression.
- **26 Triton Kernel Modules** with universal CPU/MPS fallbacks.
- **19-Function Canonical Top-Level API** (`neuralops`).
- **Full `torch.compile` Compatibility** with custom op Meta/FakeTensor registrations.
- **Serving Adapters** for vLLM and TGI runtimes (`VLLMAdapter`, `TGIAdapter`, `PagedKVCache`).
- **Zero-Shot Attention Injection** across 8 transformer architectures (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Mistral, Qwen2, T5).
- **Advanced Manifold Optimizers**: `Muon` (Newton-Schulz orthogonalization), `RiemannianStiefelOptimizer`, and `RiemannianAdam`.
- **Parametric Numerical Parity**: Validated cross-platform CPU/MPS/CUDA execution with zero skipped tests.

| Milestone Metric | v1.2.0 Baseline | v1.3.0 Delivered | Status |
| :--- | :--- | :--- | :--- |
| Canonical Attentions | 131 | 131 | **Verified 100%** |
| Compression Engines | 72 | 72 | **Verified 100%** |
| Triton Kernel Modules | 26 | 26 | **Verified 100%** |
| Dead Code LOC Pruned | 0 | 4,200+ LOC | **Pruned Cleanly** |
| Module Shadow Collisions | 3 collisions | 0 collisions | **Fully Resolved** |
| Test Parity Coverage | Partial (Mac Skipped) | 100% Parametric | **0 Failed, 0 Skipped** |

---

## v1.4.0 (Q1 2027) — Long-Context Scaling & Parallel Attention

### Strategic Focus
Scaling linear and recurrent attention mechanisms beyond 128k context lengths while preserving high distillation fidelity from pretrained dense models.

### Objectives
1. **Chunkwise Parallel Linear Attention**: Scale causal recurrent kernels across multiple GPUs using Megatron-style context parallelism for sequences exceeding 128k tokens.
2. **Selective State Scan v2**: Multi-head associative scan optimizations for Mamba-2 and Griffin architectures.
3. **Full GLUE & LRA Publication Sweeps**: Comprehensive empirical Pareto frontiers across all 131 attention mechanisms.
4. **Automated Hyperparameter Tuner**: Bayesian search integration with Ray Tune for layerwise hybrid allocation.

### Planned Deliverables
- `neuralops.parallel.ContextParallelAttention` module supporting ring-communication topologies.
- Full Long Range Arena (LRA) benchmarking scripts (ListOps, Text, Retrieval, Image, Pathfinder).
- Pre-trained checkpoints for subquadratic LLaMA-3-8B and Mistral-7B published to Hugging Face Hub.
- Fused backward Triton scan passes with memory-efficient gradient checkpointing.

| Task Item | Target Completion | Complexity | Lead Subsystem |
| :--- | :--- | :--- | :--- |
| Context Parallel Kernel Dispatch | Jan 2027 | High | `neuralops.core.triton_kernels` |
| LRA Pareto Frontier Benchmarking | Feb 2027 | Medium | `neuralops.benchmarks` |
| HF Hub Model Card Integrations | Mar 2027 | Medium | `neuralops.models` |
| Chunked GLA Multi-GPU Backward | Mar 2027 | High | `neuralops.kernels` |

---

## v1.5.0 (Q2 2027) — Hardware & Edge Expansion

### Strategic Focus
Expanding beyond NVIDIA CUDA to native Apple Metal, Intel XPU, and quantized edge runtimes.

### Objectives
1. **Intel XPU & Apple Metal Shaders**: Native Metal Performance Shaders (MPS) custom graph kernels for macOS Apple Silicon.
2. **Native vLLM C++ Custom Op Plugin**: Zero-overhead paged KV cache integration directly within the vLLM C++ runtime.
3. **Quantization-Aware Training (QAT)**: Fused FP8 and INT4 low-rank factorizations for NVIDIA Blackwell GPUs.
4. **GGUF Export Pipeline**: Standalone quantized export format for local edge execution via llama.cpp.

### Planned Deliverables
- `neuralops.export.export_gguf(...)` serialization factory.
- Metal-accelerated associative scan kernels for Apple M-series chips.
- Blackwell FP8 tensor core attention kernels targeting Hopper/Blackwell SM90/SM100 architectures.
- Low-latency edge inference harness for mobile and embedded devices.

| Task Item | Target Completion | Complexity | Lead Subsystem |
| :--- | :--- | :--- | :--- |
| Apple Metal Scan Shaders | Apr 2027 | High | `neuralops.kernels.metal` |
| Blackwell FP8 Kernel Engine | May 2027 | High | `neuralops.core.triton_kernels` |
| GGUF Model Exporter | Jun 2027 | Medium | `neuralops.export` |
| vLLM C++ Plugin Integration | Jun 2027 | Medium | `neuralops.integrations` |

---

## v2.0.0 (Q4 2027) — Major Architectural Evolution

### Strategic Focus
Next-generation asynchronous distributed model surgery and planetary-scale context inference.

### Objectives
1. **1M+ Token Sequence Parallelism**: Ring-attention hybrid linear scans spanning 64+ GPU clusters.
2. **Strict Type-Safe Specification**: Removal of legacy deprecated aliases in favor of static type checking and strict schema validation.
3. **Native Asynchronous Distributed Benchmarking**: Fault-tolerant multi-node benchmark coordinator with dynamic worker failover.
4. **Automated Neural Architecture Search (NAS)**: End-to-end evolutionary search across hybrid linear/sparse/gated layer configurations.

### Planned Deliverables
- `neuralops.distributed.RingLinearAttention` engine.
- Complete replacement of JSON configs with Pydantic v2 type-checked data models.
- Distributed benchmark telemetry coordinator with live streaming Grafana dashboards.

---

## Guiding Architectural Principles

1. **Deterministic Portability**: Every GPU-accelerated kernel must have a byte-for-byte or mathematically equivalent PyTorch fallback that executes identically on CPU and Apple Silicon.
2. **Zero Overhead Surgery**: Attention transplantation and weight surgery must never disrupt the parent model's forward signature or Hugging Face compatibility.
3. **Reproducibility First**: All benchmark runs record complete environment manifests, RNG seeds, and telemetry to immutable `.jsonl` artifacts.
4. **Zero-Stall Fallbacks**: Hardware feature absence (Triton, CUDA, AVX) must never terminate execution; the runtime must gracefully downgrade to the fastest mathematically sound alternative.

