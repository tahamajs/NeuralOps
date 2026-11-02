# Naive SVD: Plain Truncated SVD Baseline

> **Family**: SVD Spectral & Low-Rank Decomposed Attentions · **NeuralOps Registry ID**: `naive_svd`

## Executive Abstract

Classical Eckart-Young-Mirsky truncated SVD baseline without activation covariance whitening or Fisher information weighting. Serves as empirical ground truth to quantify the accuracy benefits of advanced NeuralOps compressors.

In NeuralOps, `naive_svd` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
W_r = U_{:, :r} \Sigma_{:r, :r} V_{:, :r}^T = \arg\min_{\text{rank}(M) \le r} \|W - M\|_F
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (naive_svd)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | Standard SVD truncation: O(\min(m^2 n, m n^2)) |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: CUDA C++ / Level 2: Triton JIT / Level 3: PyTorch |
| **Registry Name** | `naive_svd` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("naive_svd", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="naive_svd", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
