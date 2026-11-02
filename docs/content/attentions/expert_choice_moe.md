# Expert Choice Mixture-of-Experts (EC-MoE) Attention Block

> **Family**: Mixture-of-Experts (MoE) & Sparsely-Gated Attentions · **NeuralOps Registry ID**: `expert_choice_moe`

## Executive Abstract

Expert Choice MoE (Zhou et al., 2022) inverts traditional token-choice routing: instead of tokens selecting top-k experts, each expert independently selects its top-C tokens. This guarantees perfect load balance across experts and eliminates auxiliary load-balancing losses.

In NeuralOps, `expert_choice_moe` is integrated into the universal attention dispatch table, featuring full support for the 3-level fallback hierarchy, automatic shape validation, and end-to-end autograd backwards passes.

## Mathematical Formulation

The core operator computation is defined as:

$$
\text{Top-C tokens chosen by each expert}: \; \mathcal{T}_e = \arg\text{top-C}_{t} (X W_g)_{t, e}
$$

During execution, inputs $Q, K, V$ are projected to dimension $d$. In the causal configuration, lower-triangular masking ensures that tokens only attend to past and current time-steps, preserving autoregressive invariants.

## Architecture & Dataflow

```
Input Tokens (B, T, D)
     │
     ├───► Linear Projection (Q, K, V)
     │         │
     │         ▼
     │    Kernel / Attention Formulation (expert_choice_moe)
     │         │
     │         ▼
     └────► Scaled Output & Residual Add (B, T, D)
```

## Key Properties

| Property | Value |
|---|---|
| **Complexity** | O(E \cdot C \cdot d) perfectly load-balanced execution |
| **Causality** | Strictly Causal / Optional Bidirectional |
| **Hardware Target** | NVIDIA GPU (SM 8.0, 8.9, 9.0), CPU fallback |
| **Dispatch Level** | Level 1: CUDA C++ / Level 2: Triton JIT / Level 3: PyTorch |
| **Registry Name** | `expert_choice_moe` |

## Implementation Details

Usage via the NeuralOps Python API:

```python
import torch
from neuralops import attention, replace_attention

# Instantiate standalone operator
attn = attention("expert_choice_moe", dim=512, heads=8)
x = torch.randn(2, 64, 512, device="cuda" if torch.cuda.is_available() else "cpu")
out = attn(x)

# Or perform surgical transplant into any HuggingFace model
# replace_attention(model, attention_type="expert_choice_moe", calibrate=True)
```

## Performance & Theoretical Profile

- **Arithmetic Intensity**: High arithmetic throughput with fused SRAM tiling.
- **Memory Footprint**: Substantially lower peak VRAM relative to full quadratic attention.
- **Numerical Stability**: Accumulated in FP32 with strict dtype guarding for FP16 and BF16 execution.

---

*Part of the NeuralOps 131-attention registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
