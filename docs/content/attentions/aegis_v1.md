# Aegis V1: First-Generation Hardware-Fused Spectral Attention Kernel

> **Family**: Hardware-Fused, Low-Precision & Spectral-Transform Attentions · **NeuralOps Registry ID**: `aegis_v1`

## Executive Abstract

Aegis V1 is the founding architecture of the Aegis series in NeuralOps, executing hardware-fused spectral linear attention with low-rank singular projection and fused RMSNorm in a single CUDA execution pass.

In NeuralOps, `aegis_v1` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
\text{Aegis}_1(Q, K, V) = \mathcal{F}^{-1}\left( \mathcal{F}(Q K^T) \odot \Sigma_{\text{spectral}} \right) V
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (aegis_v1)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T d r) linear spectral complexity |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: CUDA C++ Native Extension / Level 2: Triton JIT |
| **Registry Name** | `aegis_v1` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("aegis_v1", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="aegis_v1", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
