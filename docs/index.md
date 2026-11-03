# 🚀 NeuralOps: Unified Platform for Subquadratic Attention and Compression

**NeuralOps** (`SVD_linear_Attention` **v1.6.0**) is an open-source, publication-grade deep learning acceleration platform developed at the **Department of Computer Engineering, College of Engineering, University of Tehran**, bringing together subquadratic sequence modeling, in-place neural surgery, low-rank model compression, and high-throughput inference serving.

[![Release](https://img.shields.io/badge/release-v1.6.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v1.6.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Hardware Verified](https://img.shields.io/badge/Hardware%20Verified-RTX%204070%20(SM%208.9)-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Canonical System Architecture & Invariants

NeuralOps maintains a strictly audited, immutable contract across registries, kernel dispatch hierarchies, and demonstration suites:

- **131 Attention Mechanisms (14 Architectural Families)**: Linear attention (Cosformer, Performer, GLA), state-space models (Mamba S6, SSD Mamba-2), gated linear recurrences (RetNet, RWKV-6, HGRN2), delta-rule recurrences (Gated DeltaNet, KDA), low-rank SVD projections, and hybrid architectures unified under `neuralops.core.registry.ATTENTION_REGISTRY`.
- **72 Compression Engines (8 Paradigms)**: Low-rank SVD factorization (FWSVD, AdaSVD, ASVD, Direct SVD), dynamic rank budgeting, structured pruning (SparseGPT, Wanda, ADMM, SliceGPT), subspace projection, tensor networks (Kronecker, TT), integer/floating-point quantization (FAQ, SmoothQuant, QuaRot), KV-cache optimization, and rotational transforms unified under `neuralops.core.registry.COMPRESSOR_REGISTRY`.
- **203 Unified Modules Total**: Exactly 131 attention mechanisms + 72 compression engines verified on physical hardware.
- **26 Categorized Triton Kernel Modules (62 Unified Operators)**: High-performance fused JIT operators spanning 4 architectural tiers, eliminating global memory roundtrips and executing within on-chip GPU SRAM/registers.
- **110 Canonical CUDA Modules (114 Native Implementations)**: Production-grade NVCC warp-shuffle primitives, tensor core MMA intrinsics, shared memory bank conflict elimination, and register tiling. Compatibility/support directories remain in the raw CUDA tree.
- **29 Master Jupyter Notebooks**: Complete reproducible demonstration tutorials (00–25 plus paper empirical master and integration showcases) covering end-to-end surgery, distillation healing, roofline profiling, and production serving.

---

## 3-Level Fallback Dispatch Hierarchy

Every kernel operation in NeuralOps adheres to a zero-overhead, multi-tier execution contract:

```
                  ┌─────────────────────────────────────────┐
                  │       NeuralOps Operator Dispatch       │
                  └─────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
 │    Level 1    │  Fallback    │    Level 2    │  Fallback    │    Level 3    │
 │ Native CUDA   │ ───────────► │  Triton JIT   │ ───────────► │  Vectorized   │
 │   C++ / HIP   │  (Log Warn)  │    Kernels    │  (Log Warn)  │    PyTorch    │
 └───────────────┘              └───────────────┘              └───────────────┘
  Peak Throughput                Autotuned JIT                  Universal CPU
  SM >= 7.0 / Ampere/Ada         FP16 / BF16 / FP32             Reference Parity
```

1. **Level 1 (Native CUDA C++ / HIP)**: Compiled NVCC extensions leveraging 128-bit vector loads, warp-level shuffles, and shared memory tiling across 110 canonical modules.
2. **Level 2 (Triton JIT)**: Autotuned 2D/3D tiled Triton JIT kernels compiling directly to PTX for FP16, BF16, and FP32 datatypes across 26 canonical modules (62 operators).
3. **Level 3 (Vectorized PyTorch Reference)**: Numerically validated PyTorch tensor reference guaranteeing seamless execution on CPU, Apple Silicon (MPS), and non-CUDA environments.

---

## Key Capabilities in v1.6.0

1. **One-Line In-Place Surgery**: Drop-in attention and linear layer replacement without altering downstream transformer architecture or retraining from scratch.
2. **Bit-Exact Hugging Face Roundtrip**: Zero weight drift between exported and reloaded model checkpoints via native `AutoConfig` and `AutoModel` bindings.
3. **Multi-Pillar Knowledge Distillation (KD)**: 6-pillar distillation framework (Logits KL, Feature MSE, Attention Map, Hidden Cosine, Layer Norm, Gram Matrix) recovering baseline accuracy post-surgery.
4. **Production Serving Adapters**: High-throughput integrations for vLLM (virtualized `PagedStateTable` with Copy-on-Write page isolation), Text Generation Inference (TGI continuous batching), SGLang (Radix tree prefix caching), and TensorRT-LLM.
5. **Experiment Results Ledger & Provenance Engine**: Immutable dual-storage (`.jsonl` + indexed `.sqlite`) tracking exact Git SHAs, timestamps, versions, revisions, and automated vector plotting. See [Results Ledger API](api/results_ledger.md).
6. **Interactive Visualization & Telemetry**: Live browser-based systems monitoring, WebGL 3D architecture graphs, interactive roofline profilers, and latent manifold geometry.

---

## Empirical Verification & Benchmark Grounding

All empirical benchmark claims in NeuralOps are strictly grounded in physical hardware execution on:
- **Dedicated Primary Testbed**: **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon)**: 5,888 CUDA cores, 46 SMs, 184 Tensor Cores (4th gen), 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM.
- **Data Center Secondary Testbed**: **NVIDIA H100 SXM5 (SM 90 Hopper)**. 

NeuralOps bans hardcoded numeric fallbacks and unverified placeholder outputs. The complete raw results matrix, execution logs, and LaTeX table generators are publicly accessible:

👉 **Official Hugging Face Benchmark Repository:**  
**[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch: __[Quickstart Guide](QUICKSTART.md)__  
    Install NeuralOps, execute in-place attention surgery, compress models, and deploy continuous batching serving in under 5 minutes.

-   :material-view-dashboard: __[Telemetry Dashboard](dashboard.html)__  
    Live GPU telemetry, dynamic VRAM tracking, throughput metrics, and hardware temperature monitor.

-   :material-chart-bell-curve: __[Ada Roofline Interactive](roofline_interactive.html)__  
    Explore arithmetic intensity vs. attained TFLOPS across all 26 Triton and 110 canonical CUDA modules.

-   :material-cube-scan: __[3D Attention Lab](visualizer/index.html)__  
    WebGL 3D interactive atlas inspecting the latent manifolds of 203 unified modules (131 attentions + 72 compressors). Also explore the standalone __[3D Latent Manifold](visualizer/latent_manifold.html)__.

-   :material-database-search: __[HuggingFace Benchmark Dataset](https://huggingface.co/datasets/Hooshaai/neuralops-results)__  
    Official published empirical benchmark telemetry, latency, perplexity, and error records across all 203 unified modules.

-   :material-file-pdf-box: __[Academic Theses & Manuscripts](academic_monographs.md)__  
    Direct PDF downloads: Master of Science Thesis (317 pages), Defense Slide Deck (50 slides), and NeurIPS 2026, ICLR 2027, and MLSys 2027 conference submissions.

-   :material-book-open-page-variant: __[Architecture Guide](codebase_architecture/README.md)__  
    In-depth architectural monographs on 3-level dispatch, HNC-SVD whitening, and serving virtual memory.

-   :material-code-braces: __[API Reference](api/attentions.md)__  
    Exhaustive documentation for [Attentions](api/attentions.md), [Compression](api/compression.md), [Kernels](api/kernels.md), [Serving](api/integrations.md), and [Results Ledger](api/results_ledger.md).

-   :material-history: __[Changelog](changelog.md)__  
    Release history, migration guides, and new features across all versions through v1.6.0.

</div>
