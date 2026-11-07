# NeuralOps Platform Architecture & Multi-Agent Engineering Blueprint

This directory houses the comprehensive 10-part architectural specification for the **NeuralOps v1.6.0** platform, serving as the canonical engineering blueprint for the multi-agent development and verification team.

---

## 🗺️ Master Index of Architectural Specifications & Assigned Agents

| Part | Specification Document | Subagent Assigned | Target Codebase Paths | Core Architectural Domain |
|---|---|---|---|---|
| **01** | [`01_REGISTRY_AND_LAZY_DISCOVERY.md`](01_REGISTRY_AND_LAZY_DISCOVERY.md) | `Agent 01: Registry Architect` | `neuralops/core/registry.py`, `neuralops/__init__.py` | PEP 562 Lazy Loading ($155\times$ startup acceleration), 131 Attentions & 72 Compressors |
| **02** | [`02_THREE_LEVEL_DISPATCH_HIERARCHY.md`](02_THREE_LEVEL_DISPATCH_HIERARCHY.md) | `Agent 02: Dispatch Specialist` | `neuralops/ops/cuda/_fallback.py`, `neuralops/core/backend/dispatch.py` | 3-Level Fallback Contract (CUDA $\to$ Triton $\to$ PyTorch), Zero Silent Fallback Law |
| **03** | [`03_CUDA_CPP_NATIVE_KERNEL_SUITE.md`](03_CUDA_CPP_NATIVE_KERNEL_SUITE.md) | `Agent 03: CUDA HPC Engineer` | `neuralops/core/cuda_kernels/` | 112 CUDA C++ Packages (114 Kernels), Warp Shuffles, Shared Memory Tiling ($1340.6\times$ MoE) |
| **04** | [`04_TRITON_JIT_COMPILATION_AND_AUTOTUNING.md`](04_TRITON_JIT_COMPILATION_AND_AUTOTUNING.md) | `Agent 04: Triton Specialist` | `neuralops/core/triton_kernels/` | 26 Modules (62 Operators), Autotuning Grid, Mainline Block Pointer Prohibition |
| **05** | [`05_MODEL_SURGERY_AND_INPLACE_TRANSPLANT.md`](05_MODEL_SURGERY_AND_INPLACE_TRANSPLANT.md) | `Agent 05: Surgery Specialist` | `neuralops/surgery/`, `neuralops/models/` | Recursive AST Traversal, In-Place Grafting (`replace_attention`), GQA & RoPE Reconciliation |
| **06** | [`06_HNC_SVD_AND_STREAMING_WHITENING.md`](06_HNC_SVD_AND_STREAMING_WHITENING.md) | `Agent 06: SVD Specialist` | `neuralops/compression/hnc_svd.py` | Streaming Covariance Whitening ($\mathbf{L}\mathbf{L}^\top$), Tikhonov Damping, Fisher-Ridge Projection |
| **07** | [`07_SIX_PILLAR_KNOWLEDGE_DISTILLATION.md`](07_SIX_PILLAR_KNOWLEDGE_DISTILLATION.md) | `Agent 07: Distillation Scientist` | `neuralops/trainer/distillation/` | 6-Pillar Composite KD, Spectral PiSSA Recovery, 50-Step Rapid Healing Trajectory |
| **08** | [`08_REALTIME_SERVING_AND_PAGED_KVCACHE.md`](08_REALTIME_SERVING_AND_PAGED_KVCACHE.md) | `Agent 08: Serving Architect` | `neuralops/serving/paged_kv_cache.py` | Non-Contiguous Paged Memory (`PagedKVCache`), Copy-on-Write ($7\times$ Speculative Decoding) |
| **09** | [`09_HARDWARE_TELEMETRY_AND_ROOFLINE_ENGINE.md`](09_HARDWARE_TELEMETRY_AND_ROOFLINE_ENGINE.md) | `Agent 09: Telemetry Profiler` | `neuralops/runners/monitor/` | Microsecond `CUDA Events`, SM 8.9 Roofline Analysis ($91.6\,\text{FLOPs/B}$ Ridge), Live WebSockets |
| **10** | [`10_BENCHMARK_SUITE_AND_MANIFEST_REPRODUCIBILITY.md`](10_BENCHMARK_SUITE_AND_MANIFEST_REPRODUCIBILITY.md) | `Agent 10: Benchmark Auditor` | `neuralops/benchmarks/`, `neuralops/results_saver/` | `SurgeryManifest` Provenance Ledger (SQLite/JSONL), 3-Tier Release Gates (`PASS_FAST/SLOW`) |

---

## 🔒 Platform Invariants & System Guarantees
1. **Canonical Registry Counts**: Exactly 131 Attention Mechanisms and 72 Compression Engines.
2. **Zero Silent Fallback**: Degradation to lower compute tiers is never hidden and must always log warnings and record fallback telemetry.
3. **No Fake Artifacts**: Export routines must write 100% compliant file structures or raise informative exceptions.
4. **Reproducibility Guarantee**: Every benchmark datapoint maps to an immutable Git SHA, hardware ID, and random seed.
