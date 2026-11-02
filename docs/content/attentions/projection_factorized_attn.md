# Projection Factorized SVD Linear Attention: Production-Hardened Operator

> **Family**: Multi-Head Latent & Key-Value Compressed Attentions · **NeuralOps Registry ID**: `projection_factorized_attn`

## Executive Abstract

Production-hardened factorized linear attention operator. Decouples inner rank projections across attention heads, allowing fine-grained rank allocation based on head importance metrics.

In NeuralOps, `projection_factorized_attn` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
Q = x W_{Q, 1} W_{Q, 2}, \quad K = x W_{K, 1} W_{K, 2}, \quad V = x W_{V, 1} W_{V, 2}
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (projection_factorized_attn)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(T d r) where rank $r$ is tuned per head |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: CUDA C++ / Level 2: Triton JIT / Level 3: PyTorch |
| **Registry Name** | `projection_factorized_attn` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("projection_factorized_attn", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="projection_factorized_attn", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
