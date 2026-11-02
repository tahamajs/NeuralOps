# Triton SVD Linear Attention: Fused Chunked Causal Linear Attention

> **Family**: SVD Spectral & Low-Rank Decomposed Attentions · **NeuralOps Registry ID**: `triton_svd_linear_attention`

## Executive Abstract

Triton-accelerated chunk-parallel causal linear attention specifically optimized for SVD-factorized key and value matrices. Achieves optimal arithmetic intensity on NVIDIA SM 8.9 (Ada) and SM 9.0 (Hopper).

In NeuralOps, `triton_svd_linear_attention` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
S_k = S_{k-1} + (K_{\text{chunk}})^T V_{\text{chunk}}, \quad O_{\text{chunk}} = Q_{\text{chunk}} S_{k-1} + \text{intra}(Q, K, V)
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (triton_svd_linear_attention)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T d^2) linear complexity with peak hardware efficiency |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 2: Native Triton JIT / Level 3: PyTorch reference |
| **Registry Name** | `triton_svd_linear_attention` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("triton_svd_linear_attention", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="triton_svd_linear_attention", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
