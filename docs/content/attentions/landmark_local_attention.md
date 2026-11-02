# Landmark Local Attention: Hierarchical Representative Token Pooling

> **Family**: Memory-Augmented & External Latent Recurrences · **NeuralOps Registry ID**: `landmark_local_attention`

## Executive Abstract

Inserts special landmark tokens at regular intervals that compress block-level information into compact latent anchors. Queries attend to local tokens within their block and landmark tokens across distant blocks.

In NeuralOps, `landmark_local_attention` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
L_b = \text{Landmark}(\text{Block}_b), \quad S_{ij} = \begin{cases} Q_i K_j^T & \text{if intra-block} \\ Q_i L_b^T & \text{if inter-block} \end{cases}
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (landmark_local_attention)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T B d + T (T/B) d) = O(T^{1.5} d) |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: CUDA C++ / Level 2: Triton JIT / Level 3: PyTorch |
| **Registry Name** | `landmark_local_attention` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("landmark_local_attention", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="landmark_local_attention", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
