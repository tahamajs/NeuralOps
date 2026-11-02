# Multi-Query Attention (MQA): Memory-Bandwidth Optimized Multi-Head Attention

> **Family**: Exact Softmax & IO-Aware Dense Attention · **NeuralOps Registry ID**: `mqa`

## Executive Abstract

Multi-Query Attention (MQA), introduced by Noam Shazeer (2019), drastically reduces memory bandwidth bottlenecks during autoregressive generation by sharing a single key-value head across all query heads. While standard Multi-Head Attention (MHA) loads separate K and V tensors for each head from GPU HBM, MQA broadcasts a single shared KV head across all $H$ query heads, achieving an $H\times$ reduction in memory bandwidth consumption during generation.

In NeuralOps, `mqa` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
\text{MQA}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \quad \text{where} \quad Q \in \mathbb{R}^{B \times H \times T \times d}, \; K, V \in \mathbb{R}^{B \times 1 \times S \times d}
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (mqa)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T^2 d) computation, O(T d) KV memory per sequence |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: Native CUDA C++ (Flash-MQA) / Level 2: Triton JIT / Level 3: PyTorch broadcast reference |
| **Registry Name** | `mqa` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("mqa", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="mqa", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
