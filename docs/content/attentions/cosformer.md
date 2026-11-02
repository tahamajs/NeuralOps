# Cosformer: Linear Attention with Non-Negative Cosine Feature Maps

> **Family**: Kernelized & Feature-Map Linear Attentions · **NeuralOps Registry ID**: `cosformer`

## Executive Abstract

Cosformer (Qin et al., 2022) replaces the quadratic softmax attention mechanism with a kernelized feature map based on non-negative cosine transformations. Standard linear attentions suffer from dot-product decay and poor inductive bias for local dependencies. Cosformer addresses this by combining non-negative ReLU activation with cosine-based reweighting that emphasizes proximate token interactions without sacrificing $O(N)$ linear complexity.

In NeuralOps, `cosformer` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
\phi(x) = \text{ReLU}(x) \cos\left(\frac{\pi x}{2 L}\right), \quad O = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) (\phi(K)^T \mathbf{1})}
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (cosformer)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T d^2) linear complexity, O(d^2) state space |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: Native CUDA C++ / Level 2: Triton JIT (cosformer_fused) / Level 3: Vectorized PyTorch reference |
| **Registry Name** | `cosformer` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("cosformer", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="cosformer", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
