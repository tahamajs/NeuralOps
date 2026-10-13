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
- **Serving Adapters** for vLLM and TGI runtimes.
- **Zero-Shot Attention Injection** across 8 transformer architectures (BERT, RoBERTa, DistilBERT, GPT-2, LLaMA, Mistral, Qwen2, T5).
- **Advanced Manifold Optimizers**: `Muon` (Newton-Schulz orthogonalization) and `RiemannianStiefelOptimizer`.

---

## v1.4.0 (Q1 2027) — Long-Context Scaling & Parallel Attention

### Objectives
- **Chunkwise Parallel Linear Attention**: Scale causal recurrent kernels across multiple GPUs using Megatron-style context parallelism for sequences exceeding 128k tokens.
- **Selective State Scan v2**: Multi-head associative scan optimizations for Mamba-2 and Griffin architectures.
- **Full GLUE & LRA Publication Sweeps**: Comprehensive empirical Pareto frontiers across all 131 attention mechanisms.
- **Automated Hyperparameter Tuner**: Bayesian search integration with Ray Tune for layerwise hybrid allocation.

### Planned Deliverables
- `neuralops.parallel.ContextParallelAttention` module.
- Full Long Range Arena (LRA) benchmarking scripts (ListOps, Text, Retrieval, Image, Pathfinder).
- Pre-trained checkpoints for subquadratic LLaMA-3-8B and Mistral-7B published to Hugging Face Hub.

---

## v1.5.0 (Q2 2027) — Hardware & Edge Expansion

### Objectives
- **Intel XPU & Apple Metal Shaders**: Native Metal performance shaders for macOS Apple Silicon.
- **Native vLLM C++ Custom Op Plugin**: Zero-overhead paged KV cache integration directly within the vLLM C++ runtime.
- **Quantization-Aware Training (QAT)**: Fused FP8 and INT4 low-rank factorizations for NVIDIA Blackwell GPUs.
- **GGUF Export Pipeline**: Standalone quantized export format for local edge execution via llama.cpp.

### Planned Deliverables
- `neuralops.export.export_gguf(...)` factory.
- Metal-accelerated scan kernels for Apple M-series chips.
- Blackwell FP8 tensor core attention kernels.

---

## v2.0.0 (Q4 2027) — Major Architectural Evolution

### Objectives
- **1M+ Token Sequence Parallelism**: Ring-attention hybrid linear scans.
- **Strict Type-Safe Specification**: Removal of legacy deprecated aliases in favor of static type checking.
- **Native Asynchronous Distributed Benchmarking**: Fault-tolerant multi-node benchmark coordinator.

---

## Guiding Architectural Principles

1. **Deterministic Portability**: Every GPU-accelerated kernel must have a byte-for-byte or mathematically equivalent PyTorch fallback that executes identically on CPU and Apple Silicon.
2. **Zero Overhead Surgery**: Attention transplantation and weight surgery must never disrupt the parent model's forward signature or Hugging Face compatibility.
3. **Reproducibility First**: All benchmark runs record complete environment manifests, RNG seeds, and telemetry to immutable `.jsonl` artifacts.
