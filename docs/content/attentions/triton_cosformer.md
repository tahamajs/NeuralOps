# Triton Cosformer: Fused Hardware-Accelerated Linear Attention Kernel

> **Family**: Kernelized & Feature-Map Linear Attentions · **NeuralOps Registry ID**: `triton_cosformer`

## Executive Abstract

Triton Cosformer implements a high-throughput fused 2D-tiled Triton JIT kernel executing causal Cosformer linear attention. Operates with chunk-parallel associative scans within streaming multiprocessor (SM) shared memory, completely eliminating high-bandwidth memory (HBM) intermediate round trips.

In NeuralOps, `triton_cosformer` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
S_t = S_{t-1} + K_t^T V_t, \quad O_t = \frac{Q_t S_t}{Q_t Z_t} \quad \text{fused in SRAM tile} \; [B_T \times d]
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (triton_cosformer)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T d^2) compute, zero intermediate HBM allocation |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 2: Native Triton JIT / Level 3: PyTorch fallback |
| **Registry Name** | `triton_cosformer` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("triton_cosformer", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="triton_cosformer", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
