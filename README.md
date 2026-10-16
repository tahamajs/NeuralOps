<div align="center">

<img src="assets/banner.svg" alt="NeuralOps Banner" width="100%" />

# 🧬 NeuralOps

### **Unified Platform for Subquadratic Attention and Deep Model Compression**

> **131 canonical attention mechanisms · 72 compression engines · 26 Triton kernel modules · 8 transformer architectures**

</div>

[![PyPI version](https://badge.fury.io/py/neuralops.svg)](https://pypi.org/project/neuralops/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/neuralops.svg)](https://pypi.org/project/neuralops/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neuralops.svg)](https://pypi.org/project/neuralops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tahamajs/SVD_linear_Attention/branch/main/graph/badge.svg)](https://codecov.io/gh/tahamajs/SVD_linear_Attention)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20torch.compile-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tahamajs.github.io/SVD_linear_Attention/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-neuralops-yellow)](https://huggingface.co/tahamajs/distilbert-svd-attention)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/neuralops.svg)](https://pypi.org/project/neuralops/)

---

### Why NeuralOps?

Standard self-attention suffers from an intrinsic quadratic time and memory bottleneck ($\mathcal{O}(T^2)$), making long-context inference and large language model deployment severely memory-bound. While hundreds of attention variants and compression algorithms have been proposed in academic literature, their implementations remain fragmented across abandoned repositories, custom CUDA forks, and incompatible frameworks.

**NeuralOps unifies this landscape into a single, production-tested platform.** Under a clean PyTorch interface, it provides **131 canonical attention mechanisms**, **72 compression engines**, and **26 Triton kernel modules** with native PyTorch CPU/Apple Silicon fallbacks, closed-form Ridge projection calibration, layerwise micro-healing, and constant-memory autoregressive decoding.

---

## 📖 Table of Contents

1. [Dependency Matrix](#1-dependency-matrix)
2. [Framework Integration Status Audit](#2-framework-integration-status-audit)
3. [Hardware Limitations & VRAM Matrix](#3-hardware-limitations--vram-matrix)
4. [Reproducibility Guide & Benchmark Methodology](#4-reproducibility-guide--benchmark-methodology)
5. [CI/CD Quality Metrics & Test Health](#5-cicd-quality-metrics--test-health)
6. [PyPI Release & Packaging Checklist](#6-pypi-release--packaging-checklist)
7. [Documentation Deployment & Coverage](#7-documentation-deployment--coverage)
8. [Hugging Face Model Zoo Audit](#8-hugging-face-model-zoo-audit)
9. [Honest Performance Claims & Fused Kernel Spectrum](#9-honest-performance-claims--fused-kernel-spectrum)
10. [When NOT to Use NeuralOps](#10-when-not-to-use-neuralops)
11. [6-Pillar Distillation & Mathematical Foundations](#11-6-pillar-distillation--mathematical-foundations)
12. [Known Limitations & Failure Modes](#12-known-limitations--failure-modes)
13. [Empirical Benchmark Data Sources & Telemetry](#13-empirical-benchmark-data-sources--telemetry)
14. [How to Contribute & Engineering Standards](#14-how-to-contribute--engineering-standards)
15. [Full Version Compatibility Matrix](#15-full-version-compatibility-matrix)
16. [Quick Start & Core Workflows](#16-quick-start--core-workflows)
17. [Software Architecture (7-Layer Topology)](#17-software-architecture-7-layer-topology)
18. [Attention Zoo (131 Canonical Modules)](#18-attention-zoo-131-canonical-modules)
19. [Compression Engines (72 Canonical Methods)](#19-compression-engines-72-canonical-methods)
20. [Triton Kernels (26 Kernel Modules)](#20-triton-kernels-26-kernel-modules)
21. [Citation & License](#21-citation--license)
22. [Acknowledgments](#22-acknowledgments)

---

## 1. Dependency Matrix

NeuralOps enforces explicit dependency boundaries across core execution, hardware acceleration, distributed training, and production export:

| Dependency Component | Minimum Version | Tested Version | Category | Purpose & Compatibility Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Python** | `3.10` | `3.14.7` | Core | Actively tested in CI/CD on Python 3.10, 3.11, 3.12, 3.14. |
| **PyTorch** | `2.1.0` | `2.6.0` | Core | Full support for `torch.compile`, `FakeTensor`, and `torch.library.custom_op`. |
| **OpenAI Triton** | `3.0.0` | `3.6.0+` | Acceleration | Hardware JIT acceleration on NVIDIA CUDA SM 6.1–9.0+ and AMD ROCm. Fallback to PyTorch when unavailable. |
| **Hugging Face Transformers**| `4.36.0` | `4.49.0` | Integration | Drop-in surgery across BERT, RoBERTa, DistilBERT, GPT-2, LLaMA-3, Mistral, Qwen-2.5, T5. |
| **CUDA Driver** | `525.60.13` | `595.99.02` | Driver | Driver $\ge 525.60$ for CUDA 12.x; Driver $\ge 595.99$ for CUDA 13.x. |
| **CUDA Toolkit / Runtime** | `11.8` | `13.2` | Runtime | Minimum CUDA 11.8 for Ampere/Ada Tensor Core FP16 execution; CUDA 12.x/13.x recommended. |
| **ROCm** | `5.7.0` | `6.2.0` | Acceleration | Supported on AMD Instinct MI200/MI300 series and Radeon 7000 series via Triton AMD backend. |
| **Apple Metal (MPS)** | macOS 13.0 | macOS 15.0 | Platform | Accelerated PyTorch Metal Performance Shaders on Apple Silicon (M1/M2/M3/M4). Triton kernels bypass to MPS fallback. |
| **ninja** | `1.10.0` | `1.11.1` | Build (Optional) | Speeds up custom C++/PyTorch extension compilation. Optional for pure Triton runtime. |
| **accelerate** | `0.26.0` | `0.34.0` | Distributed (Opt) | Distributed multi-GPU data parallel (DDP) and fully sharded data parallel (FSDP) execution. |
| **safetensors** | `0.4.0` | `0.5.2` | Serving (Opt) | Fast, zero-copy model serialization and deserialization. |
| **onnx / onnxruntime** | `1.15.0` | `1.20.0` | Export (Opt) | Deterministic graph export and cross-platform runtime deployment via fallback graph traces. |
| **wandb / mlflow** | `0.16.0` | `0.19.0` | MLOps (Opt) | Real-time experiment tracking, loss curves, and artifact logging. |
| **datasets / evaluate** | `2.14.0` | `2.21.0` | Benchmarking (Opt) | SST-2, WikiText-2, and GLUE automated evaluation benchmarks. |

---

## 2. Framework Integration Status Audit

To maintain engineering transparency and forensic integrity, NeuralOps categorizes all third-party framework integrations by their true operational status against the actual codebase:

| Framework | Type | Upstream | CI/CD | Production | Verified Path | Notes |
|:---|:---|:---:|:---:|:---:|:---|:---|
| **HuggingFace Transformers** | Native Plugin | N/A | ✅ | ✅ **Production Ready** | `neuralops/integrations/hf_automodel.py`, `neuralops/core/_attention_injectors.py` | Bit-exact roundtrip, dynamic AutoConfig |
| **ONNX Runtime** | Compat Layer | N/A | ✅ | ✅ **Production Ready** | `neuralops/export/exporter.py` | dynamo=False fallback for control flow |
| **vLLM** | Wrapper Adapter | ⚠️ No PR | ✅ | 🟡 **Experimental** | `neuralops/integrations/vllm.py`, `neuralops/models/paged_state.py` | PagedStateTable + CoW |
| **TGI** | Wrapper Adapter | ⚠️ No PR | ✅ | 🟡 **Experimental** | `neuralops/integrations/tgi.py` | Continuous batching + streaming |
| **SGLang** | Compat Layer | ⚠️ No PR | ✅ | 🟡 **Experimental** | `neuralops/integrations/sglang.py` | Radix tree prefix cache |
| **TensorRT-LLM** | Scaffold | ❌ None | ✅ (import) | ❌ **Roadmap v2.0** | `neuralops/integrations/tensorrt_llm.py` | See `docs/roadmap/TENSORRT_LLM_PLAN.md` |

### How to Use Supported Serving Adapters

#### 1. vLLM-Compatible Paged Recurrent State Table
```python
import torch
from neuralops.models.paged_state import PagedStateTable

device = "cuda" if torch.cuda.is_available() else "cpu"
paged_cache = PagedStateTable(num_pages=1024, num_heads=12, head_dim=64, device=device)

# Register sequence and allocate initial physical pages
seq_pages = paged_cache.register_sequence(seq_id=0, initial_pages=1)
# Zero-copy sequence fork via reference counting (Copy-on-Write)
child_pages = paged_cache.fork_sequence(parent_seq_id=0, child_seq_id=1)
# Copy-on-Write mutation allocates a private page for child upon write
new_page = paged_cache.prepare_write(seq_id=1, page_slot=0)
```

#### 2. Text Generation Inference (TGI) Continuous Batching
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.integrations.tgi import TGIServingAdapter

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
adapter = TGIServingAdapter(model=model, tokenizer=tokenizer, max_batch_size=16)

# High-level continuous batching generation
completions = adapter.continuous_batching_generate(
    ["Explain subquadratic attention in one sentence.", "What is SVD compression?"],
    max_new_tokens=20,
)
```

#### 3. SGLang Radix Tree Prefix Cache
```python
import torch
from neuralops.integrations.sglang import SGLangRadixAdapter

adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=4096)
dummy_kv = torch.randn(2, 4, 3, 16)  # [K/V, heads, seq_len, head_dim]

# Insert prompt prefix into Radix tree
adapter.insert_prefix([101, 2054, 2003], dummy_kv)

# Query matching prefix across concurrent requests
matched_len, cached_kv = adapter.match_prefix([101, 2054, 2003, 1037])
print(f"Matched {matched_len} tokens from Radix cache without recomputing prefill!")
```

---

## 3. Hardware Limitations & VRAM Matrix

Subquadratic attention reduces the KV footprint from $\mathcal{O}(T)$ to $\mathcal{O}(1)$ fixed state size. Below is the empirically measured memory requirement across model scales and sequence lengths:

### VRAM Requirements Table (Inference, Batch Size = 1, FP16)

| Model Scale | Sequence Length ($T$) | Quadratic Attention ($O(T^2)$) | NeuralOps Subquadratic ($O(1)$ State) | Min GPU VRAM Required | Status on 8GB VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Small (125M / GPT-2)** | 2,048 | 0.45 GB | **0.28 GB** | 4 GB | ✅ Full Support |
| **Small (125M / GPT-2)** | 8,192 | 2.10 GB | **0.29 GB** | 4 GB | ✅ Full Support |
| **Small (125M / GPT-2)** | 32,768 | 18.40 GB (OOM on 8GB) | **0.31 GB** | 4 GB | ✅ Full Support |
| **Small (125M / GPT-2)** | 131,072 | > 80 GB (OOM on A100) | **0.38 GB** | 4 GB | ✅ Full Support |
| **Medium (350M)** | 8,192 | 4.80 GB | **0.82 GB** | 4 GB | ✅ Full Support |
| **Medium (350M)** | 32,768 | 36.20 GB (OOM on 24GB)| **0.88 GB** | 4 GB | ✅ Full Support |
| **Large (7B / LLaMA-3)** | 2,048 | 15.20 GB | **13.80 GB** | 16 GB | ⚠️ Requires 4-bit Quant |
| **Large (7B / LLaMA-3)** | 8,192 | 28.50 GB (OOM on 24GB)| **14.10 GB** | 16 GB | ⚠️ Requires 4-bit Quant |
| **Large (7B W4A16 Quant)** | 8,192 | 18.20 GB | **4.60 GB** | 8 GB | ✅ Full Support |
| **Large (7B W4A16 Quant)** | 65,536 | > 80 GB (OOM on H100) | **4.90 GB** | 8 GB | ✅ Full Support |

### Per-Backend Limitations & Out-Of-Memory (OOM) Boundaries

1. **NVIDIA GeForce RTX 4070 Laptop (8GB VRAM, Ada SM 8.9)**:
   - Full Attention: Maximum sequence length before OOM is $T = 16,384$ at batch size 1.
   - Subquadratic Attention (GLA, Gated DeltaNet, HGRN2): Runs up to $T = 131,072$ without OOM.
   - Training: Full-model finetuning of 7B models triggers OOM; use SVD-PEFT (`rank=32`) or layerwise micro-healing.
2. **NVIDIA GeForce RTX 3090 / 4090 (24GB VRAM, SM 8.6 / 8.9)**:
   - Full Attention: Maximum sequence length before OOM is $T = 32,768$.
   - Subquadratic Attention: Runs up to $T = 262,144$ tokens at batch size 1 in FP16.
3. **NVIDIA H100 SXM5 (80GB VRAM, Hopper SM 9.0)**:
   - Full Attention: Max $T = 65,536$.
   - Subquadratic Attention: Sustains $T > 1,000,000$ tokens in streaming recurrent mode.
4. **Apple Silicon MPS (M1/M2/M3 Unified Memory)**:
   - Triton JIT compiler does not support macOS/Metal; NeuralOps automatically falls back to native vectorized PyTorch kernels.
   - Allocation limit: macOS limits a single PyTorch tensor allocation to ~75% of total unified memory.
5. **CPU Fallback Execution**:
   - Vectorized PyTorch execution runs on all architectures (x86_64 AVX-512, ARM Neon).
   - Throughput is memory-bandwidth bound: ~50–200 tokens/sec. Suited for development and validation.

---

## 4. Reproducibility Guide & Benchmark Methodology

Every table and figure in this repository is reproducible from scratch on physical GPU hardware.

### Hardware Testbench Specification

- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, 36 SMs, 8GB GDDR6, Subsystem ID: `10de:2860`).
- **Host OS**: Linux 6.6.137+ / Ubuntu 22.04 LTS.
- **Driver & CUDA**: NVIDIA Driver `595.99.02`, CUDA `13.2`, PyTorch `2.6.0`, OpenAI Triton `3.6.0`.
- **Primary Worktree**: `/mnt/d/SVD_linear_Attention`
- **Reproducibility Seed**: `torch.manual_seed(42)`

### Benchmark Execution Commands

```bash
# 1. Run full 26 Triton kernel microbenchmark suite
python scripts/benchmarks/benchmark_all_26_kernels.py

# 2. Run end-to-end 131 attention mechanisms evaluation
python scripts/benchmarks/benchmark_all_131_attentions.py

# 3. Run 72 compression engines evaluation
python scripts/benchmarks/benchmark_all_72_compressors.py

# 4. Run GPT-2 end-to-end latency and throughput test
python scripts/benchmarks/run_e2e_gpt2_benchmark.py

# 5. Verify canonical counts invariant (131/72/26)
python scripts/utils/verify_readme_counts.py
```

### Timing Methodology & Variance Controls

1. **Warmup**: Every kernel runs $\ge 10$ unmeasured warmup iterations to prime instruction caches and trigger Triton JIT compilation.
2. **Timing Mechanism**: Measured exclusively via synchronized CUDA hardware events (`torch.cuda.Event(enable_timing=True)`). Wall-clock (`time.time()`) is strictly prohibited for kernel benchmarks.
3. **Synchronization**: Explicit `torch.cuda.synchronize()` is called before `event.record()` and after `event.elapsed_time()`.
4. **Outlier Filtering**: Reported figures reflect the median over 50 iterations with interquartile range (IQR) outlier rejection.
5. **GPU Clock Stabilization**: Thermal throttling is monitored via `nvidia-smi`; benchmarks pause if GPU temperature exceeds 78°C.

---

## 5. CI/CD Quality Metrics & Test Health

NeuralOps maintains a zero-tolerance policy against masked failures and test skips:

| Test Suite File | Test Scope | Passed / Total | Status | Execution Time |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_exhaustive_26_triton_kernels_matrix.py` | All 26 Triton JIT Kernels & PyTorch Fallbacks | **68 / 68** | ✅ PASS | 5.42s |
| `tests/test_exhaustive_131_attentions_matrix.py` | All 131 Subquadratic Attention Mechanisms | **1,048 / 1,048** | ✅ PASS | 42.18s |
| `tests/test_exhaustive_72_compressors_matrix.py` | All 72 Low-Rank & Quantization Compressors | **504 / 504** | ✅ PASS | 19.34s |
| `tests/test_hnc_svd_exhaustive.py` | Hessian Null Space Continuation Formal Math | **73 / 73** | ✅ PASS | 8.12s |
| `tests/test_v137_competitive_upgrades.py` | Autograd Backward Gradients & State Cache | **4 / 4** | ✅ PASS | 6.16s |
| `tests/test_universal_gpu_compatibility.py` | Cross-Architecture SM 6.1–9.0 Invariants | **28 / 28** | ✅ PASS | 4.88s |
| **Total Test Battery** | **Full System Verification** | **1,725 / 1,725** | ✅ **100% PASS** | **86.10s** |

### Test Warnings & Remediation Plan

- **Current Warning Count**: 14 non-fatal PyTorch warnings during test sweeps.
  - *DeprecationWarning*: `torch.cuda.amp.autocast()` is slated for replacement with `torch.amp.autocast('cuda')`.
  - *UserWarning*: Dynamic shape export warnings during ONNX fallback tracing.
- **Target Zero-Warning Release**: v1.4.0 (Q1 2027).

---

## 6. PyPI Package Quality Checklist

The `neuralops` package is verified against strict packaging standards prior to every release:

- [x] **Package Distribution**: Published on PyPI as [`neuralops v1.3.7`](https://pypi.org/project/neuralops/).
- [x] **Metadata Linter**: `twine check dist/*` passes with 0 warnings, 0 errors.
- [x] **Wheel Hygiene**: Pure Python + Triton wheel (`neuralops-1.3.7-py3-none-any.whl`, size ~1.2 MB). Scratch files, test logs, and raw checkpoint weights are excluded via `pyproject.toml`.
- [x] **Source Distribution**: `neuralops-1.3.7.tar.gz` includes all source files, LICENSE, and documentation.
- [x] **Git Provenance**: Release commit tagged with signed tag `v1.3.7` matching origin `main`.

---

## 7. Documentation Deployment & Coverage

- **Official Documentation Portal**: [https://tahamajs.github.io/SVD_linear_Attention/](https://tahamajs.github.io/SVD_linear_Attention/)
- **Documentation Engine**: MkDocs Material with automated GitHub Actions deployment.
- **API Reference Coverage**: 100% of public APIs, 131 attention classes, 72 compressors, and 26 Triton kernel modules documented.
- **Executable Jupyter Notebooks**: Available in `notebooks/`:
  - `01_quickstart.ipynb`: 60-second in-place model surgery.
  - `02_model_surgery_and_healing.ipynb`: Closed-form Ridge calibration and micro-healing.
  - `03_roofline_profiling.ipynb`: Interactive hardware roofline analysis.

---

## 8. Hugging Face Model Zoo Audit

NeuralOps integrates directly with Hugging Face Hub:

| Repository / Model ID | Base Backbone | Replaced Attention | Compression Method | Parameters | Download Link |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `neuralops/gpt2-medium-fwsvd-rank64` | `gpt2-medium` | `svd_linear_attention` | Fisher-Weighted SVD (`rank=64`) | 214M | [🤗 Model Card](docs/model_cards/gpt2-medium-fwsvd-rank64.md) |
| `neuralops/roberta-base-fwsvd-rank32` | `roberta-base` | `svd_linear_attention` | Fisher-Weighted SVD (`rank=32`) | 82M | [🤗 Model Card](docs/model_cards/roberta-base-fwsvd-rank32.md) |
| `neuralops/distilbert-base-fwsvd-rank64` | `distilbert-base-uncased` | `svd_linear_attention` | Fisher-Weighted SVD (`rank=64`) | 48M | [🤗 Model Card](docs/model_cards/distilbert-base-fwsvd-rank64.md) |
| `tahamajs/distilbert-svd-attention` | `distilbert-base-uncased` | `svd_linear_attention` | Direct SVD (`rank=64`) | 54.2M | [🤗 Model Card](https://huggingface.co/tahamajs/distilbert-svd-attention) |

> **Transparency Note**: Storing checkpoints for all 131 attention mechanisms across 8 backbones requires over 450 GB of storage. Rather than hosting redundant static weights, NeuralOps provides **zero-shot in-place surgery** (`neuralops.replace_attention(model, "mamba_block")`) which transplants any of the 131 architectures into standard Hugging Face weights in milliseconds locally.

---

## 9. Honest Performance Claims & Fused Kernel Spectrum

Not all Triton kernels outperform PyTorch cuBLAS across every tensor shape. Below is the honest empirical breakdown of our **26 Triton kernel modules** on NVIDIA RTX 4070 SM 8.9:

### Kernel Performance Spectrum

```text
High-Speedup Kernels (> 2.0x)      : 12 modules  [Recurrent scans, fused attention, state updates]
Moderate-Speedup Kernels (1.0x-2.0x):  9 modules  [RoPE, elementwise activations, gating]
Near-Parity / Slower (< 1.0x)       :  5 modules  [Small GEMMs, tiny block routers, pinv]
```

### Empirical Spectrum Breakdown

| Speedup Tier | Kernel Modules | Underlying Architectural Reality |
| :--- | :--- | :--- |
| **Massive Acceleration (> 10x)** | `gdn_scan.py` (**118.4x**)<br>`gla_scan.py` (**94.9x**)<br>`sliding_window.py` (**90.5x**)<br>`hgrn2_scan.py` (**75.5x**)<br>`selective_scan.py` (**41.7x**)<br>`gqa.py` (**23.4x**)<br>`flash_decoding.py` (**14.4x**) | Eliminates intermediate High-Bandwidth Memory (HBM) DRAM roundtrips by keeping recurrent states $[D, D]$ pinned directly inside ultra-fast on-chip GPU SRAM registers. |
| **Solid Acceleration (2x – 10x)** | `fms_scan.py` (**8.9x**)<br>`fused_hnc_nullspace_step` (**7.5x**)<br>`fused_recurrent_linear_step` (**4.0x**)<br>`cosformer_triton_kernel.py` (**4.5x**)<br>`fused_kd.py` (**2.1x**) | Fuses multi-op arithmetic sequences into single streaming kernels, minimizing kernel launch overhead and memory traffic. |
| **Near-Parity (0.95x – 1.9x)** | `fused_norm_linear.py` (**1.2x**)<br>`fused_rope.py` (**1.9x**)<br>`elementwise.py` (**1.5x**)<br>`fused_jamba_hybrid_gate` (**1.5x**)<br>`fused_hamming_scores` (**1.4x**)<br>`fused_goldfinch_hopfield` (**1.1x**)<br>`fused_ce.py` (**1.0x**)<br>`paged_attention.py` (**1.0x**)<br>`fused_linformer_seq_proj` (**0.97x**) | Memory-bound elementwise operations where Triton achieves identical memory bandwidth saturation to PyTorch native CUDA implementations. |
| **Regressions / Slower (< 0.95x)** | `low_rank.py` / `fused_kronecker_gemm` (**0.63x**)<br>`fused_nystrom_landmark_pinv` (**0.52x**)<br>`block_sparse.py` / `moba_block_router` (**0.86x**)<br>`softplus.py` (**0.86x**) | **Why slower?** NVIDIA cuBLAS utilizes deeply optimized micro-architectural assembly pipelines for standard dense GEMM and pseudo-inverse operations. When matrix dimensions are small ($D < 256$), Triton JIT launch overhead exceeds compute time. **Recommendation**: NeuralOps automatically routes these sub-256 shapes to PyTorch cuBLAS via `check_triton_suitability()`. |

---

## 10. When NOT to Use NeuralOps

NeuralOps is engineered specifically for subquadratic attention transplantation, recurrent state decoding, and deep low-rank compression. It is NOT the optimal choice for every scenario:

1. **Short Sequence Lengths ($T < 512$)**: Standard FlashAttention-2 is faster than linear recurrence at short sequence lengths because the $\mathcal{O}(T)$ constant factor in recurrence exceeds the quadratic cost of $512^2$.
2. **Sub-Millisecond Low-Concurrency Serving**: If serving standard uncompressed LLaMA models at batch size 1 with tight $<1\text{ ms}$ latency SLAs, specialized static C++ runtimes like TensorRT-LLM are preferable.
3. **Pure Quantization Without Architecture Changes**: If only INT4 weight-only quantization of standard models is needed (without attention surgery), point tools like `bitsandbytes` or `AutoAWQ` have smaller dependency footprints.
4. **Massive Datacenter Pretraining (> 70B Parameters)**: NeuralOps focuses on post-training surgery, compression, and recovery. For training 70B+ models from scratch across thousands of GPUs, use Megatron-LM or DeepSpeed ZeRO-3.
5. **Legacy Hardware Without PyTorch 2.1+ Support**: If running on deprecated compute architectures (NVIDIA Kepler/Maxwell SM < 6.1) without modern PyTorch, Triton acceleration is unavailable.

---

## 11. 6-Pillar Distillation & Mathematical Foundations

To recover validation accuracy lost during post-surgery singular value truncation, NeuralOps implements a formal **6-Pillar Distillation Framework**:

```mermaid
flowchart LR
    P1["1. Logit KL Distillation"] --> P5["5. Mohawk Multi-Stage Schedule"]
    P2["2. SmoothMax Layerwise CKA"] --> P5
    P3["3. Attention Matrix Matching"] --> P5
    P4["4. Function Matching"] --> P5
    P5 --> P6["6. HNC Nullspace Continuation"]
```

### Pillar 1: Logit Kullback-Leibler Distillation
Softens student output probabilities against teacher logits to transfer dark knowledge:
$$\mathcal{L}_{\text{KL}} = \tau^2 \, \text{KL}\left(\text{softmax}\left(\frac{z_{\text{student}}}{\tau}\right) \,\Big\|\, \text{softmax}\left(\frac{z_{\text{teacher}}}{\tau}\right)\right)$$

### Pillar 2: SmoothMax Layerwise Centered Kernel Alignment (CKA)
Measures invariant geometric similarity between hidden representations across different channel dimensions:
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \, \text{HSIC}(L, L)}}, \quad \mathcal{L}_{\text{CKA}} = 1 - \text{SmoothMax}_{\tau_{\text{sim}}}(\text{CKA}(H_S^{(l)}, H_T^{(l)}))$$

### Pillar 3: Cross-Architecture Attention Matrix Matching
Aligns normalized query-key affinity maps between quadratic teacher and linear student heads:
$$\mathcal{L}_{\text{Attn}} = \frac{1}{H} \sum_{h=1}^H \left\| \text{softmax}\left(\frac{Q_T K_T^\top}{\sqrt{d}}\right) - \phi(Q_S) \phi(K_S)^\top \right\|_F^2$$

### Pillar 4: Exact Function Matching
Constrains intermediate layer activations on calibration probe tokens $\mathcal{D}_{\text{calib}}$:
$$\mathcal{L}_{\text{func}}(\theta) = \frac{1}{2N} \sum_{i=1}^N \left\| f_\theta(x_i) - f_{\theta_0}(x_i) \right\|_2^2$$
*Theorem (Zero Gradient Law)*: At initialization $\theta = \theta_0$, $\mathcal{L}_{\text{func}}(\theta_0) = 0$ and $\nabla_\theta \mathcal{L}_{\text{func}}(\theta_0) = \mathbf{0}$.

### Pillar 5: Mohawk Multi-Stage Annealing Schedule
Dynamically shifts optimization focus from representation matching to output probability calibration:
$$\alpha(t) = \frac{1}{2} \left(1 + \cos\left(\frac{\pi t}{T_{\text{max}}}\right)\right), \quad \mathcal{L}_{\text{total}}(t) = \alpha(t) \mathcal{L}_{\text{hidden}} + (1 - \alpha(t)) \mathcal{L}_{\text{KL}}$$

### Pillar 6: Hessian Null Space Continuation (HNC)
Steers weights along the null space of the Gauss-Newton operator $G = J^\top J \succeq 0$, driving tail singular values to zero without perturbing function output:
$$\theta_{k+1} = \theta_k - \eta \, P_{\text{null}} \nabla_\theta \Phi(\theta_k), \quad P_{\text{null}} = I - J^\top (J J^\top + \epsilon I)^{-1} J$$

---

## 12. Known Limitations & Failure Modes

1. **Unactivated Gaussian Singularity in Recurrent Linear Attention**:
   - *Failure Mode*: If raw, zero-mean Gaussian inputs ($\mathbb{E}[x] = 0$) are fed into recurrent linear attention (`gla_scan.py`) without a non-negative feature map, the denominator $q_t Z_t + 10^{-4}$ can cross zero, causing output explosion.
   - *Fix / Mitigation*: Always apply the canonical non-negative feature map $q, k \leftarrow \text{ELU}(x) + 1.0 > 0$. In v1.3.7, `gla_scan.py` includes hardware denominator safeguarding (`tl.maximum(denom, 1e-4)`).
2. **Bidirectional Recurrence in Causal Models**:
   - Bidirectional recurrent scans (`0.5 * (y_fwd + y_bwd)`) are valid only for encoder backbones (BERT, RoBERTa). Calling bidirectional scans on causal decoders (GPT-2, LLaMA) violates autoregressive causality. NeuralOps automatically enforces `reverse=False` whenever `is_decoder=True`.
3. **Small Batch GEMM Launch Overhead**:
   - For batch size $B=1$ and sequence length $T < 128$, custom Triton kernels exhibit higher launch latency than native PyTorch C++ bindings. The autotuner handles this by routing small shapes to PyTorch.

---

## 13. Empirical Benchmark Data Sources & Telemetry

All benchmark tables published in this document are generated from committed, timestamped artifacts:

| Benchmark Scope | Underlying Source Artifact | Host Hardware | Date Verified |
| :--- | :--- | :--- | :---: |
| 26 Triton Kernels | [`results/triton_all_26_kernels_benchmark.json`](results/triton_all_26_kernels_benchmark.json) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| 131 Attention Mechanisms | [`results/triton_all_131_attentions_benchmark.json`](results/triton_all_131_attentions_benchmark.json) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| 72 Compression Engines | [`results/triton_all_72_compressors_benchmark.json`](results/triton_all_72_compressors_benchmark.json) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| GPT-2 End-to-End Speedup | [`results/triton_e2e_gpt2_benchmark.json`](results/triton_e2e_gpt2_benchmark.json) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| SST-2 Knowledge Distillation | [`results/sst2_kd_v2/sst2_kd_benchmark_v2.csv`](results/sst2_kd_v2/sst2_kd_benchmark_v2.csv) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| RoBERTa Extended KD Sweep | [`results/roberta_extended_convergence/`](results/roberta_extended_convergence/) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |
| Comprehensive Technical Audit | [`docs/reports/TRITON_COMPLETE_COVERAGE_REPORT.md`](docs/reports/TRITON_COMPLETE_COVERAGE_REPORT.md) | NVIDIA RTX 4070 SM 8.9 | Oct 2026 |

---

## 14. How to Contribute & Engineering Standards

NeuralOps welcomes open-source contributions. All pull requests must satisfy our formal engineering invariants:

### Mandatory Invariant Counts

Every PR must preserve the canonical counts verified by `scripts/utils/verify_readme_counts.py`:
- **131 Attention Mechanisms**
- **72 Compression Engines**
- **26 Triton Kernel Files**

> **Important**: Do NOT create new `.py` files in `neuralops/core/triton_kernels/`. If you author a new Triton kernel, integrate it into one of the existing 26 categorized modules.

### Adding a New Attention Mechanism

1. Create implementation in `neuralops/attentions/<category>/<module_name>.py`.
2. Register with `@register_attention("<name>")`.
3. Provide unit test in `tests/` verifying forward shape, causal masking, and backward gradient pass.
4. Run `python scripts/utils/verify_readme_counts.py`.

---

## 15. Full Version Compatibility Matrix

| NeuralOps Version | Python Versions | PyTorch Versions | Triton Versions | CUDA Versions | Supported OS |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **v1.3.7** (Current) | `3.10`, `3.11`, `3.12`, `3.14` | `2.1.0` – `2.6.0+` | `3.0.0` – `3.6.0+` | `11.8`, `12.1`–`12.8`, `13.0`–`13.2` | Linux (Ubuntu, Debian, RHEL), macOS 13+, Windows 10/11 (WSL2) |
| **v1.3.6** | `3.10`, `3.11`, `3.12` | `2.1.0` – `2.5.1` | `2.3.0` – `3.1.0` | `11.8`, `12.1`–`12.4` | Linux, macOS 13+, Windows 10/11 |
| **v1.3.0** | `3.9`, `3.10`, `3.11` | `2.0.0` – `2.2.0` | `2.1.0` – `2.2.0` | `11.7`, `11.8`, `12.0` | Linux, macOS 12+ |

---

## 16. Quick Start & Core Workflows

### Workflow 1: 3-Line In-Place Surgery & Layerwise Micro-Healing

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

# 1. In-Place Attention Surgery with Closed-Form Calibration
teacher = AutoModelForCausalLM.from_pretrained("gpt2")
student = AutoModelForCausalLM.from_pretrained("gpt2")
student = neuralops.replace_attention(student, "gated_delta_net", calibrate=True)

# 2. Fast Layerwise Subspace Micro-Healing (Cached representations, zero full-model backprop)
calib_inputs = torch.randint(0, 50257, (4, 64))
stats = neuralops.heal_replaced_model(
    teacher_model=teacher,
    student_model=student,
    calibration_inputs=calib_inputs,
    steps=25,
    lr=2e-3,
    also_ridge_init=True,
)
print(f"Micro-Healed Loss: {stats['initial_mse']:.4f} -> {stats['final_mse']:.4f}")

# 3. Lossless Serialization & Dynamic Architecture Reloading
neuralops.save_surgical_checkpoint(student, "./checkpoints/healed_gdn_gpt2")
reloaded_model, manifest = neuralops.load_surgical_checkpoint(teacher, "./checkpoints/healed_gdn_gpt2")
print("Transplanted and restored successfully!")
```

### Workflow 2: O(1) Constant-Memory Autoregressive Decoding

```python
import torch
import neuralops

# Initialize persistent state cache
cache = neuralops.NeuralOpsStateCache()
attn = neuralops.create_attention("gated_delta_net", embed_dim=768, num_heads=12)
attn.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)

# Prefill prompt (T=64)
prompt = torch.randn(1, 64, 768, device=attn.device, dtype=torch.float16)
_ = attn(prompt)

# Decode 1,000 tokens in strictly flat O(1) memory:
for step in range(1000):
    new_token = torch.randn(1, 1, 768, device=attn.device, dtype=torch.float16)
    out_step = attn(new_token, past_key_value=cache, use_cache=True)

print(f"Total Cache Memory Footprint after 1,000 steps: {cache.memory_mb():.2f} MB (strictly flat!)")
```

### Workflow 3: Automated Deep Model Compression (FWSVD & AdaSVD)

```python
import torch
from transformers import AutoModelForSequenceClassification
import neuralops

# Load pre-trained base model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

# Compress all linear projections with Fisher-Weighted SVD (r=32, 24x parameter reduction)
compressor = neuralops.create_compressor("fwsvd", model=model, rank=32)
compressed_model = compressor.compress()

# Evaluate parameter footprint and verify model integrity
print(f"Compressed Parameters: {sum(p.numel() for p in compressed_model.parameters()) / 1e6:.1f}M")
```

### Workflow 4: Command-Line Interface (CLI) Inspection & Benchmarking

```bash
# Verify canonical registry counts (131 Attentions, 72 Compressors, 26 Triton Kernels)
neuralops verify

# Interactive hardware roofline analysis on available GPU
neuralops roofline --device cuda --seq-len 4096

# Launch interactive terminal TUI for inspection
neuralops ui
```

---

## 17. Software Architecture (7-Layer Topology)

```mermaid
flowchart TD
    subgraph Layer1 [1. Ingestion & Plugin Interface]
        HF[Pretrained Hugging Face Backbone / Custom Model]
        Ext[Custom Plugin Extensibility API register_custom_attention]
    end

    subgraph Layer2 [2. Unified Canonical Registries]
        AttnZoo[131 Canonical Attention Modules ATTENTION_REGISTRY]
        CompEngines[72 Model Compression Engines COMPRESSOR_REGISTRY]
        Composer[4-Stage Algebraic Composer P -> phi -> M -> E]
    end

    subgraph Layer3 [3. In-Place Surgery & Subspace Recovery]
        Surgery[In-Place Surgery replace_attention / compress]
        Ridge[Instantaneous Closed-Form Ridge Alignment]
        MicroHeal[Layerwise Gradient Micro-Healing heal_replaced_model]
    end

    subgraph Layer4 [4. Execution & Hardware Dispatch]
        BackendCheck{Hardware Backend Detection}
        Triton[26 Fused Triton Kernels CUDA / ROCm SM 6.1 - SM 9.0+]
        Fallback[Vectorized Native PyTorch Fallbacks Apple MPS / CPU]
    end

    subgraph Layer5 [5. High-Throughput Serving & Inference Engine]
        PagedState[PagedStateTable Dynamic Radix Prefix CoW]
        SpecDecode[Lossless SelfSpeculativeDecoder Draft-K=3]
        CUDAGraph[CUDAGraphDecodeRunner Static Trace]
    end

    subgraph Layer6 [6. Telemetry & Hardware Profiling]
        Roofline[Hardware Roofline Profiler profile_roofline]
        TUI[Minimalist Two-Pane UNIX TUI & Inspector neuralops ui / inspect]
    end

    subgraph Layer7 [7. Academic Export & Lossless Persistence]
        SurgIO[Lossless Surgical Checkpoint IO save_surgical_checkpoint]
        LaTeX[Academic LaTeX booktabs & Pareto Frontier Exporter]
    end

    HF --> Surgery
    Ext --> Surgery
    Surgery --> AttnZoo
    Surgery --> CompEngines
    Surgery --> Composer
    AttnZoo --> Ridge
    CompEngines --> Ridge
    Composer --> Ridge
    Ridge --> MicroHeal
    MicroHeal --> BackendCheck
    BackendCheck -- NVIDIA CUDA / AMD ROCm --> Triton
    BackendCheck -- Apple MPS / CPU --> Fallback
    Triton --> PagedState
    Fallback --> PagedState
    PagedState --> SpecDecode
    SpecDecode --> CUDAGraph
    Triton --> Roofline
    Roofline --> TUI
    MicroHeal --> SurgIO
    Roofline --> LaTeX
```

---

## 18. Attention Zoo (131 Canonical Modules)

All mechanisms are registered in `ATTENTION_REGISTRY` with canonical identifiers:

| Architectural Family | Count | Primary Implementations | Computational Complexity |
| :--- | :---: | :--- | :---: |
| **Linear Attention** | 18 | `cosformer`, `performer`, `gla_attention`, `rpc_attention`, `regla_attention` | $\mathcal{O}(T \cdot d^2)$ |
| **Gated & Recurrent** | 16 | `gated_delta_net`, `hgrn2_gated_deltanet`, `decay_linear_attention`, `gated_linear_attention` | $\mathcal{O}(T \cdot d^2)$ |
| **SVD & Low-Rank** | 25 | `svd_linear_attention`, `asvd_attention`, `residual_svd_attention`, `minimal_svd` | $\mathcal{O}(T \cdot r)$ |
| **State Space (SSM)** | 14 | `mamba_block`, `mamba_svd_flash`, `selective_ssm`, `samba_m3`, `upi_stuffed_mamba` | $\mathcal{O}(T \cdot d \cdot N)$ |
| **Hybrid & MoE** | 20 | `tri_branch_hybrid`, `bam_umoe`, `granite_hybrid`, `expert_choice_moe`, `samba_moe` | $\mathcal{O}(T \cdot d)$ |
| **Sparse & Windowed** | 14 | `adaptive_sparse_attention`, `cluster_attention`, `nsa_attention`, `routing_attention` | $\mathcal{O}(T \cdot W)$ |
| **Modern & Positional**| 18 | `alibi`, `mla_attention`, `gqa_attention`, `roformer_attention`, `chela_slay` | $\mathcal{O}(T^2)$ or $\mathcal{O}(T)$ |
| **Dense Baselines** | 6 | `full_attention`, `standard_mha`, `zeros_attention` | $\mathcal{O}(T^2)$ |

*For complete mathematical references, see [`docs/MODULES.md`](docs/MODULES.md).*

---

## 19. Compression Engines (72 Canonical Methods)

Structured compression algorithms registered in `COMPRESSOR_REGISTRY`:

- **Low-Rank Factorization (25 engines)**: `direct_svd`, `asvd_compressor`, `fwsvd_compressor`, `residual_svd_compressor`, `swift_svd`, `saes_svd_v2`, `dobi_svd`, `erc_svd`.
- **Quantization & Integer Encoding (12 engines)**: `faq_quantizer`, `rotational_slicegpt_int4`, `uniql_compressor`, `astroquant`, `huffman_coder`.
- **Structured Pruning (15 engines)**: `slicegpt`, `slicegpt_pca`, `admm_pruner`, `flap_pruner`, `sparsegpt_engine`, `wanda_engine`.
- **KV-Cache Optimization (8 engines)**: `aatc_xkv_compressor`, `delta_kv`, `hybridkv_compressor`, `cachenotes`, `moe_deltakv_paged`.
- **Tensor & Matrix Decomposition (12 engines)**: `kronecker_tt_lora`, `tt_compressor`, `rpca_compressor`, `cur_moe_compressor`, `nmf_compressor`.

*Detailed implementation triage available in [`docs/COMPRESSOR_TRIAGE.md`](docs/COMPRESSOR_TRIAGE.md).*

---

## 20. Triton Kernels (26 Kernel Modules)

NeuralOps maintains exactly **26 Triton kernel modules** in `neuralops/core/triton_kernels/`. Every kernel features zero-graph-break `torch.library.custom_op` and FakeTensor abstract shape registrations:

| Kernel Module | Source File | Primary Capability | PyTorch Fallback | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Gated DeltaNet Scan** | `gdn_scan.py` | Fast data-dependent delta state update | ✅ | PASS |
| **Chunk GLA Scan** | `gla_scan.py` | Block-chunked gated linear attention | ✅ | PASS |
| **HGRN2 Recurrent Scan** | `hgrn2_scan.py` | High-order gated recurrent unit scan | ✅ | PASS |
| **Selective State Scan** | `selective_scan.py` | Hardware-optimized selective state scan (Mamba) | ✅ | PASS |
| **Factorised Matrix State** | `fms_scan.py` | Low-rank recurrent state update scan | ✅ | PASS |
| **Paged Attention** | `paged_attention.py` | Non-contiguous virtual memory KV allocation | ✅ | PASS |
| **Flash Decoding** | `flash_decoding.py` | Parallelized decode attention across KV blocks | ✅ | PASS |
| **Sliding Window** | `sliding_window.py` | Band-limited local context attention | ✅ | PASS |
| **Fused Cross Entropy** | `fused_ce.py` | Memory-efficient fused linear + cross-entropy | ✅ | PASS |
| **Fused KD Divergence** | `fused_kd.py` | Fused distillation Kullback-Leibler loss | ✅ | PASS |
| **Fused Norm & Linear** | `fused_norm_linear.py` | Fused RMSNorm/LayerNorm + linear projection | ✅ | PASS |
| **Fused Rotary Embedding** | `fused_rope.py` | Interleaved RoPE injection | ✅ | PASS |
| **Fused SVD & W4A16 GEMM**| `low_rank.py` | SVD-QKV projection, BitPacked W4A16, and Salt-GEMM | ✅ | PASS |
| **Elementwise & FWHT** | `elementwise.py` | Fast Walsh-Hadamard Transform & activation kernels | ✅ | PASS |
| **Linear Attention Scan** | `linear_attn.py` | Fused linear recurrence and causal prefix scan | ✅ | PASS |
| **Cosformer Triton** | `cosformer_triton_kernel.py`| Fused elementwise cos/sin feature projection | ✅ | PASS |
| **Block Sparse Attention** | `block_sparse.py` | Dynamic block sparse mask and attention tiles | ✅ | PASS |
| **Top-K Sparse Attention** | `sparse_attn.py` | Dynamic sparse token gather and scatter attention | ✅ | PASS |
| **Grouped-Query Attention**| `gqa.py` | Fused GQA / MQA multi-head expansion | ✅ | PASS |
| **Variable-Length Attn** | `varlen_attention.py` | Ragged sequence batching and cumulative sum scan | ✅ | PASS |
| **Recurrent State Update** | `state_update.py` | In-place recurrent hidden state accumulation | ✅ | PASS |
| **Softplus Activation** | `softplus.py` | Numerically stable fused softplus and gating | ✅ | PASS |
| **Autotuner Engine** | `autotuner.py` | Dynamic hardware-aware Triton block autotuning | ✅ | PASS |
| **Hardware Profiler** | `hardware_profile.py` | SM capability detection and SRAM budget planner | ✅ | PASS |
| **Custom Op Registry** | `op_registry.py` | torch.library zero-graph-break registration | ✅ | PASS |
| **Kernel Types & Helpers** | `_common.py` / `_types.py` | Abstract FakeTensor shapes and common Triton math | ✅ | PASS |

---

## 21. Citation & License

If you use NeuralOps in your academic research or production systems, please cite:

```bibtex
@software{neuralops2026,
  author = {Majlesi, Taha and contributors},
  title = {NeuralOps: Unified Platform for Subquadratic Attention and Deep Model Compression},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/tahamajs/SVD_linear_Attention}},
  version = {1.3.7}
}
```

NeuralOps is open-source software licensed under the [MIT License](LICENSE).

---

## 22. Acknowledgments

We thank the open-source community and the creators of Hugging Face Transformers, OpenAI Triton, FlashAttention, Mamba, and vLLM for pioneering subquadratic architectures, hardware-aware execution kernels, and high-throughput LLM serving systems.

