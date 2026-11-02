# How NeuralOps Works: Complete Systems Architecture Guide

> An in-depth technical monograph detailing the execution semantics, 3-level kernel dispatch hierarchy, attention surgical transplant engine, and 72-engine model compression pipeline.

---

## 1. Executive Architecture Overview

NeuralOps is a unified deep learning systems framework designed to solve the two biggest operational bottlenecks of modern Transformer models:
1. **The Quadratic Context Bottleneck**: Standard softmax attention computes full $N \times N$ Gram matrices, consuming $\mathcal{O}(N^2)$ memory and compute bandwidth.
2. **The Memory-Bandwidth Serving Bottleneck**: Autoregressive decoding spends up to 90% of execution time streaming key-value matrices from high-bandwidth GPU memory (HBM).

NeuralOps addresses these challenges through a tightly integrated three-layer architecture:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User & Python Client Layer                      │
│   from neuralops import attention, replace_attention, compress        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 In-Place Surgical & Compression Engine                 │
│  - Architecture Inspection & Topological Dependency Scanner            │
│  - 131 Attention Transplants with Dynamic Normalization & KV Rewriting │
│  - 72 Compression Engines (SVD, Pruning, Quantization, Distillation)   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  Zero-Overhead Dispatch Coordinator                    │
│             Level 1: Native CUDA C++ (128-bit vector loads)            │
│             Level 2: Autotuned Triton JIT (SRAM chunk-parallel)        │
│             Level 3: Vectorized PyTorch Reference (Universal Fallback) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 3-Level Fallback Dispatch Hierarchy

Every kernel operation in NeuralOps implements a formal 3-level fallback contract with zero silent degradation:

### Level 1: Native CUDA C++ / HIP Extension
- **Target**: Maximum theoretical memory bandwidth and compute throughput on NVIDIA Ada Lovelace (SM 8.9), Hopper (SM 9.0), and Blackwell (SM 10.0).
- **Techniques**: Warp-shuffle intrinsics (`__shfl_xor_sync`), 128-bit vectorized global memory transactions (`float4`), register-resident matrix tiles, and bank-conflict-free shared memory padding.
- **Packages**: 130 native CUDA C++ kernel packages.

### Level 2: Autotuned Triton JIT Kernels
- **Target**: High-productivity hardware compilation without manual C++ toolchain compilation.
- **Techniques**: 2D-tiled block iterators, chunk-parallel associative Blelloch scans in SRAM, automatic grid heuristic autotuning across tile sizes `[BLOCK_M, BLOCK_N]`.
- **Invariants**: Strict ban on `tl.make_block_ptr` / `tl.advance` (to ensure universal compatibility across standard Triton versions; Huawei Ascend NPU exempt).
- **Modules**: 26 canonical modules housing 62 unified operators across 4 tiers.

### Level 3: Vectorized PyTorch Reference
- **Target**: Universal verification, unit testing, CPU execution, and Apple Silicon MPS support.
- **Semantics**: Pure PyTorch tensor operations preserving exact mathematical parity within `atol=1e-3, eps=1e-5`.

```python
# Forcing backends explicitly for benchmarking
from neuralops import force_kernel_backend

# Force pure Triton JIT execution
force_kernel_backend("triton")

# Force native CUDA C++ execution
force_kernel_backend("cuda")

# Reset to automatic hierarchical dispatch
force_kernel_backend(None)
```

---

## 3. Attention Surgery: In-Place Attention Transplant

The Attention Surgery engine allows researchers and practitioners to replace the attention mechanism of any pretrained PyTorch or HuggingFace model with any of the **131 registered attention architectures** in two lines of Python:

```python
from transformers import AutoModelForCausalLM
from neuralops import replace_attention, inspect_architecture

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 1. Inspect layer shapes and identify candidate modules
spec = inspect_architecture(model)

# 2. Perform surgical transplant
replace_attention(
    model=model,
    attention_type="cosformer",  # Any of 131 registered mechanisms
    calibrate=True,              # Automatic activation covariance calibration
    target_layers=None           # None = replace all attention layers
)
```

### What Happens During Surgery:
1. **Module Discovery**: Graph scanner recursively identifies all self-attention and cross-attention blocks.
2. **Weight Projection Adaptation**: Automatically maps input projection dimensions ($W_q, W_k, W_v, W_o$) into the target attention's required shapes.
3. **KV Cache Rewriter**: Reconfigures the model's autoregressive cache buffers (e.g. converting multi-head cache to compact recurrent state or multi-query format).
4. **Causal Mask Binding**: Installs strictly causal lower-triangular or chunked scan operators to prevent temporal token leakage.

---

## 4. The 72-Engine Compression Pipeline

NeuralOps houses 72 compression algorithms categorized into five distinct paradigms:

| Paradigm | Engines | Key Representative | Typical Ratio | Perplexity Retention |
|---|---|---|---|---|
| **SVD & Low-Rank** | 24 | HNC-SVD, FWSVD, SVD-LLM | 2.0x – 4.0x | 97% – 99.5% |
| **Structured Pruning** | 16 | ProxSparse, ADMM Channel, 2:4 Sparsity | 1.5x – 2.5x | 94% – 98% |
| **Quantization** | 14 | Spectron-FP8, MX-FP4, SmoothQuant | 2.0x – 4.0x | 98% – 99.8% |
| **KV-Cache Pruning** | 10 | AATC-XKV, StreamingLLM Sinks, H2O | 3.0x – 8.0x | 95% – 99% |
| **Rotational & Manifold** | 8 | SliceGPT-PCA, Stiefel Riemannian SVD | 1.3x – 2.0x | 96% – 98.5% |

### Activation Covariance Whitening (HNC-SVD):
Rather than truncating weights via naive Frobenius norm minimisation $\|W - M\|_F$, NeuralOps whitens weights using the empirical activation covariance matrix $\mathbf{\Sigma}_X = \mathbb{E}[X^T X]$:

$$
\min_{A, B} \| (W - A B) \mathbf{\Sigma}_X^{1/2} \|_F^2
$$

This guarantees that the low-rank approximation error is suppressed along the principal activation directions traversed by real data.

---

## 5. Post-Surgery Knowledge Distillation Healing

Following surgery or compression, models undergo 6-pillar knowledge distillation healing to recover truncated spectral energy:

$$
\mathcal{L}_{\text{heal}} = w_1 \mathcal{L}_{\text{CE}}(y_s, y_t) + w_2 \mathcal{L}_{\text{KL}}(p_s, p_t) + w_3 \mathcal{L}_{\text{hidden}}(h_s, h_t) + w_4 \mathcal{L}_{\text{cosine}}(h_s, h_t)
$$

Trained under the **Warmup-Stable-Decay (WSD)** learning rate scheduler:
1. **Warmup** (first 10% steps): Linear ramp to prevent gradient shock on transplanted weights.
2. **Stable** (next 70% steps): High constant learning rate exploring the loss basin.
3. **Decay** (final 20% steps): Cosine annealing driving weights into flat, generalizing minima.

---

## 6. Export and Production Serving

Optimized models can be exported to all major production runtimes:
- **PyTorch SafeTensors**: Native weights with custom dispatcher.
- **ONNX Runtime**: Graph export with fused SRAM kernels.
- **GGUF v3**: Fully quantized binary container for edge deployment via llama.cpp.
- **vLLM / SGLang**: Zero-overhead RadixAttention and PagedAttention KV-cache integration.

---

*NeuralOps Architectural Specification · [Back to Hub](../../index.html)*
