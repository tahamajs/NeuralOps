# NeuralOps Platform Architecture & Multi-Agent Engineering Blueprint (v2.1.0)

This directory houses the comprehensive 10-part architectural specification for the **NeuralOps v2.1.0** platform, serving as the canonical engineering blueprint for the multi-agent development and verification team.

---

## 🏛️ The 7-Layer Architecture Overview

NeuralOps is structured into 7 strictly segregated execution layers enforcing unidirectional dependencies:

```text
Layer 7: Provenance, Schema Validation & Cryptographic Persistence (SafeTensors, 5-Format Persistence)
   ▲
Layer 6: Hardware Telemetry, Microsecond Profiling & Roofline Diagnostics (ColdCacheFlusher, Events)
   ▲
Layer 5: High-Throughput Serving, Inference & AOT Offline Teacher Caching (PagedKVCache, 0 MB VRAM)
   ▲
Layer 4: In-Place Neural Surgery, Modular Surgical Deltas & Composite Adaptation (ModularDelta, 6-Pillar KD)
   ▲
Layer 3: Vectorized Numerical Reference Implementations (PyTorch Golden Standards)
   ▲
Layer 2: Native Triton JIT Acceleration (58 Tiled Kernels without Block Pointer Deprecations)
   ▲
Layer 1: Native CUDA C++ Microarchitectural Kernels (140 Packages, Warp Shuffles, 128-bit Loads)
```

---

## 🗺️ Master Index of Architectural Specifications & Assigned Agents

| Part | Specification Document | Subagent Assigned | Target Codebase Paths | Core Architectural Domain |
|---|---|---|---|---|
| **01** | [`01_REGISTRY_AND_LAZY_DISCOVERY.md`](01_REGISTRY_AND_LAZY_DISCOVERY.md) | `Agent 01: Registry Architect` | `neuralops/core/registry.py`, `neuralops/__init__.py` | PEP 562 Lazy Loading ($155\times$ startup acceleration), 131 Attentions & 72 Compressors |
| **02** | [`02_THREE_LEVEL_DISPATCH_HIERARCHY.md`](02_THREE_LEVEL_DISPATCH_HIERARCHY.md) | `Agent 02: Dispatch Specialist` | `neuralops/ops/cuda/_fallback.py`, `neuralops/core/backend/dispatch.py` | 3-Level Fallback Contract (CUDA $\to$ Triton $\to$ PyTorch), Zero Silent Fallback Law |
| **03** | [`03_CUDA_CPP_NATIVE_KERNEL_SUITE.md`](03_CUDA_CPP_NATIVE_KERNEL_SUITE.md) | `Agent 03: CUDA HPC Engineer` | `neuralops/ops/cuda/` | 140 CUDA C++ Packages (114+ Kernels), Warp Shuffles, Shared Memory Tiling |
| **04** | [`04_TRITON_JIT_COMPILATION_AND_AUTOTUNING.md`](04_TRITON_JIT_COMPILATION_AND_AUTOTUNING.md) | `Agent 04: Triton Specialist` | `neuralops/ops/triton/` | 58 Tiled Triton Kernels (26 Modules), Autotuning Grid, Mainline Block Pointer Prohibition |
| **05** | [`05_MODEL_SURGERY_AND_INPLACE_TRANSPLANT.md`](05_MODEL_SURGERY_AND_INPLACE_TRANSPLANT.md) | `Agent 05: Surgery Specialist` | `neuralops/surgery/`, `neuralops/models/` | Recursive AST Traversal, In-Place Grafting, Modular Surgical Deltas (97% reduction) |
| **06** | [`06_HNC_SVD_AND_STREAMING_WHITENING.md`](06_HNC_SVD_AND_STREAMING_WHITENING.md) | `Agent 06: SVD Specialist` | `neuralops/compression/hnc_svd.py` | Streaming Covariance Whitening ($\mathbf{L}\mathbf{L}^\top$), Tikhonov Damping, Fisher-Ridge Projection |
| **07** | [`07_SIX_PILLAR_KNOWLEDGE_DISTILLATION.md`](07_SIX_PILLAR_KNOWLEDGE_DISTILLATION.md) | `Agent 07: Distillation Scientist` | `neuralops/trainer/distillation/` | 6-Pillar Composite KD, AOT Offline Teacher Caching (0 MB VRAM, 2x Throughput) |
| **08** | [`08_REALTIME_SERVING_AND_PAGED_KVCACHE.md`](08_REALTIME_SERVING_AND_PAGED_KVCACHE.md) | `Agent 08: Serving Architect` | `neuralops/serving/paged_kv_cache.py` | Non-Contiguous Paged Memory (`PagedKVCache`), Copy-on-Write ($7\times$ Speculative Decoding) |
| **09** | [`09_HARDWARE_TELEMETRY_AND_ROOFLINE_ENGINE.md`](09_HARDWARE_TELEMETRY_AND_ROOFLINE_ENGINE.md) | `Agent 09: Telemetry Profiler` | `neuralops/runners/monitor/` | Microsecond `CUDA Events`, SM 8.9 Roofline Analysis ($91.6\,\text{FLOPs/B}$ Ridge), ColdCacheFlusher |
| **10** | [`10_BENCHMARK_SUITE_AND_MANIFEST_REPRODUCIBILITY.md`](10_BENCHMARK_SUITE_AND_MANIFEST_REPRODUCIBILITY.md) | `Agent 10: Benchmark Auditor` | `neuralops/benchmarks/`, `neuralops/results_saver/` | `SurgeryManifest` Provenance Ledger (5 formats: JSON/JSONL/CSV/MD/Parquet), Quality Gates |

---

## 🔒 Platform Invariants & System Guarantees
1. **Canonical Registry Counts**: Exactly 131 Attention Mechanisms and 72 Compression Engines.
2. **Zero Silent Fallback**: Degradation to lower compute tiers is never hidden and must always log warnings and record fallback telemetry.
3. **No Fake Artifacts**: Export routines must write 100% compliant file structures or raise informative exceptions.
4. **Strict Module Boundaries (AGENTS.md §7)**: Low-level ops and layers never import runners, CLI, or reporting; results savers never mutate models.
5. **Offline Teacher Representation Caching (AGENTS.md §21.1)**: Teacher forward passes are precomputed AOT to `/dev/shm`, ensuring 0 MB teacher GPU VRAM allocation.
6. **Modular Surgical Delta Protocol (AGENTS.md §21.2)**: Model surgeries export only isolated modified weights (`attention_weights.safetensors`), reducing storage by $\ge 97\%$.
7. **Reproducibility Guarantee**: Every benchmark datapoint maps to an immutable Git SHA, hardware ID, and random seed.
