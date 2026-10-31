# NeuralOps Strategic Roadmap

This document outlines the architectural roadmap, development milestones, and release strategy for **NeuralOps** (`SVD_linear_Attention`).

---

## Current Release: v1.7.1 (October 2026) — Core Hardening, Diagnostics & Defect Remediation

The v1.7.1 patch release remediates all non-FLA audit findings, establishes honest function naming contracts, eliminates silent exception swallowing in core paths, and aligns repository telemetry:

- **Honest Operator Naming**: Replaced pseudo-Triton function names in `ops/triton/low_rank.py` with explicit `_pytorch` routines, maintaining deprecated `_triton` aliases with transparent warning emissions.
- **Fail-Closed Silent Exception Remediation**: Converted bare `except Exception: pass` swallows in registry and metric resolution to structured warning diagnostics and typed exception propagation.
- **112 Package Canonical Invariant**: Unified all repository telemetry to declare strictly 112 CUDA kernel package directories housing 114 native implementations.
- **Python Compatibility Target Alignment**: Harmonized linter targets (`ruff.toml`, `pyproject.toml`) to `py310` to guarantee strict runtime compatibility with `requires-python = ">=3.10"`.
- **v1.7.0 Retrospective & Partial Delivery Scope**: Formalized v1.7.0 delivery status (36.7% delivered: 4 major Triton scans [GLA, RetNet, GDN, HGRN2], `neuralops.pipeline()`, one-line `compress()`, `transplant()`, `export()` APIs, PagedStateTable, and UniversalAttentionWrapper signature caching). Explicitly deferred native CUDA FLA kernels, Hopper TMA, Blackwell TMEM, and Ascend 910B ops to v1.8.0.

| Milestone Metric | v1.6.0 Baseline | v1.7.0 Delivered | v1.7.1 Current | Status |
| :--- | :--- | :--- | :--- | :--- |
| Canonical Attentions | 131 | 131 (`neuralops.layers`) | 131 (`neuralops.layers`) | **Verified 100%** |
| Compression Engines | 72 | 72 (`neuralops.compression`) | 72 (`neuralops.compression`) | **Verified 100%** |
| Triton Kernel Modules | 26 (62 Ops) | 33 modules (FLA scans added) | 33 modules / 62 ops | **Verified 100%** |
| Native CUDA Packages | 110 | 112 packages (114 kernels) | 112 packages (114 kernels) | **Verified 100%** |
| High-Level Pipeline | None | `neuralops.pipeline` | `neuralops.pipeline` | **Verified 100%** |
| Dispatch Mode Caching | Dynamic search | `_cached_dispatch_mode` | `_cached_dispatch_mode` | **Verified 100%** |
| Pseudo-Triton Naming | Legacy `_triton` | Legacy `_triton` | `_pytorch` + aliases | **Honest Naming** |
| Target Python Target | Inconsistent | `py314` vs `py312` | `py310` unified | **Unified Target** |

---

## v1.8.0 (Q1 2027) — Flash Linear Attention (FLA) & Advanced Hardware Acceleration

### Strategic Focus
Native Flash Linear Attention (FLA) ecosystem integration, native CUDA FLA chunkwise kernels, and specialized hardware extensions.

### Objectives
1. **Optional FLA Integration**: Native adapter bridges for `fla-linear-attention` under the isolated `fla:*` registry namespace (`pip install 'neuralops[fla]'`).
2. **Native CUDA FLA Extensions**: Fused chunkwise CUDA kernels for GLA, Gated DeltaNet, RetNet, Based, and RWKV6 (`neuralops/ops/cuda/`).
3. **Advanced Microarchitecture Headers**: Hopper TMA (`_hopper_tma.cuh`) and Blackwell TMEM (`_blackwell_tmem.cuh`).
4. **Distributed Scale Primitives**: Context parallelism, FSDP2, and Megatron-LM sequence-parallel hooks.

---

## v2.0.0 (Q3 2027) — Native Serving Hooks & Edge Expansion

### Strategic Focus
Direct C++ runtime scheduler integration for high-throughput enterprise LLM serving engines.

### Objectives
1. **vLLM PagedAttention Integration**: Native scheduler page allocator and dynamic KV cache write lifecycle integration.
2. **SGLang RadixAttention Integration**: Deep C++/Triton dispatch hooks for Radix Trie prefix caching.
3. **TensorRT-LLM Custom Engine Plugin**: Direct compilation of hybrid subquadratic transformer layers into serialized TRT execution plans.
4. **TGI Continuous Batching Hooks**: Native Rust/C++ request slotting and streaming decoders.
