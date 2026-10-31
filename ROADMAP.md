# NeuralOps Strategic Roadmap

This document outlines the architectural roadmap, development milestones, and release strategy for **NeuralOps** (`SVD_linear_Attention`).

---

## Current Release: v1.9.0 (October 2026) — Surgery Framework Foundation

NeuralOps v1.9.0 ships 5 surgery operators (2 stable, 3 experimental) under `neuralops.surgery`, including attention replacement and MLP shrinkage, alongside fail-closed stubs for INT4 quantization and distillation.

The v1.8.1 release consolidates ecosystem stability following the v1.7.0 partial delivery (36.7% across 6 work packages) and v1.8.0 Flash Linear Attention (`fla`) integration:

- **v1.8.1**: Post-audit stabilization — renamed pseudo-Triton operations to `*_pytorch` with deprecated aliases, eliminated silent exception swallows across data paths, normalized CUDA kernel counts (112 packages / 114 native implementations), and removed stale telemetry artifacts.
- **v1.8.0**: Integrated upstream Flash Linear Attention (`fla`) via optional, fail-closed extension bridge (`neuralops.integrations.fla`) with strict return-arity negotiation, isolated extension registry, and zero disruption to canonical invariants.
- **v1.7.0**: Partial delivery (36.7%) — advanced low-rank kernels, cache bridge primitives, and initial serving shims.
- **v1.6.0**: Tier modernization, PEP 562 lazy loading, and CPU-safe execution architecture.

- **131 Attention Mechanisms** physically migrated to `neuralops.layers` across 12 mathematical families.
- **72 Model Compression Engines** physically migrated to `neuralops.compression` spanning SVD, low-rank, pruning, and quantization.
- **26 Triton Kernel Modules** (housing 62 unified Triton JIT operators across 4 tiers) in `neuralops.ops.triton` with universal CPU fallbacks.
- **114 Native CUDA C++ Extensions** across 112 packages in `neuralops.ops.cuda` targeting Ada Lovelace (SM 8.9), Hopper (SM 90), and Blackwell (SM 100).
- **PEP 562 Lazy Loading Architecture**: Bare import `import neuralops` executes in `<30ms` using standard library only, deferring PyTorch/accelerator loading until modules are accessed.
- **Consolidated Primitives**: Single canonical implementations for `SVDLinear`, `LoRALinear`, `FactorizedLinear`, `NeuralOpsConfig`, and `compute_flops_estimate`.
- **Decoupled Telemetry & Benchmarking**: Heavy plotting and analysis dependencies (`matplotlib`, `pandas`) relocated out of core to `neuralops.analysis` and `neuralops.benchmarks`.
- **Silent Failure Elimination & Honest Stubs**: Replaced bare exception swallows with explicit diagnostics; honest `IntegrationNotImplementedError` stubs for third-party serving backends (vLLM, SGLang, TRT-LLM, TGI).
- **CPU Distributed Test Suite**: Added `tests/distributed/test_distributed_cpu.py` covering tensor/context parallelism with zero SIGABRT/crash risks on macOS and CPU-only systems.
- **29 Master Jupyter Notebooks** in `notebooks/`.

| Milestone Metric | v1.5.0 Baseline | v1.6.0 Delivered | Status |
| :--- | :--- | :--- | :--- |
| Canonical Attentions | 131 | 131 (`neuralops.layers`) | **Verified 100%** |
| Compression Engines | 72 | 72 (`neuralops.compression`) | **Verified 100%** |
| Triton Kernel Modules | 26 (62 Ops) | 26 (62 Ops) (`ops.triton`) | **Verified 100%** |
| Native CUDA Packages | 110 | 112 packages (114 kernels) | **Verified 100%** |
| Bare Import Latency | ~1.5s+ | < 30ms (PEP 562) | **Sub-30ms Verified** |
| Dead Code LOC Pruned | Baseline | 0 Dead Code Issues | **0 AST Issues** |
| CPU Test Health | Partial collection | 48 passed, 0 failed | **100% Passing** |

---

## v1.7.0 (Q1 2027) — Long-Context Scaling & Chunkwise Parallelism

### Strategic Focus
Scaling linear and recurrent attention mechanisms beyond 128k context lengths while preserving high distillation fidelity from pretrained dense models.

### Objectives
1. **Chunkwise Parallel Linear Attention**: Scale causal recurrent kernels across multiple GPUs using context parallelism for sequences exceeding 128k tokens.
2. **Selective State Scan v2**: Multi-head associative scan optimizations for Mamba-2 and Griffin architectures.
3. **Automated Hyperparameter Tuner**: Layerwise hybrid attention allocation search.

---

## v2.0.0 (Q3 2027) — Native Serving Hooks & Edge Expansion

### Strategic Focus
Transitioning serving integration stubs to native runtime C++ hooks and kernel extensions:
1. **Native vLLM Custom Op Plugin**: In-tree custom kernel registration with vLLM PagedAttention v2.
2. **TensorRT-LLM Plugin**: Custom TRT plugin compilation for fused recurrent attention kernels.
3. **Quantized Edge Runtimes**: Quantized INT4/FP8 linear attention execution for Apple Silicon and edge devices.
