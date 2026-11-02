# SliceGPT PCA Compressor: Principal Component Analysis Weight Slicing

> **Paradigm**: Rotational & Structural Slicing · **NeuralOps Registry ID**: `slicegpt_pca`

## Executive Abstract

SliceGPT-PCA computes orthogonal transformation matrices from activation principal components, rotating the weight space such that entire rows and columns can be physically removed without retarding dense GEMM speed.

Integrated into the NeuralOps compression engine suite, `slicegpt_pca` operates seamlessly across standard HuggingFace, PyTorch, and GGUF model formats, enabling high-ratio parameter compression with verified accuracy retention.

## Mathematical Formulation

The optimization and decomposition objective is formulated as:

$$
W' = Q^T W P, \quad \text{Slice}(W') = W'_{:r, :r} \quad \text{where} \; Q, P \text{ are orthogonal PCA rotation matrices}
$$

Where weights are decomposed or pruned to minimize task loss degradation under specified rank or sparsity constraints.

## Algorithmic Workflow

```
Original Dense Model
     │
     ├───► 1. Profile Layer Activation Spectra / Covariance
     │
     ├───► 2. Optimize Objective & Formulate Decomposed Layers
     │
     ├───► 3. Substitute Factorized Modules into Model Graph
     │
     └───► 4. Post-Surgery Knowledge Distillation / Healing
```

## Key Properties

| Property | Value |
|---|---|
| **Compression Ratio** | 1.3x - 2.0x structural parameter reduction |
| **Execution Speed** | Fast (orthogonal PCA rotation) |
| **Target Layers** | `nn.Linear`, Attention Projections, FFN Blocks |
| **Export Formats** | PyTorch, SafeTensors, ONNX, GGUF v3 |
| **Registry Name** | `slicegpt_pca` |

## Usage

```python
import torch
from neuralops import compress

# Configure compression engine
engine = compress("slicegpt_pca", rank_ratio=0.5)

# Execute compression on target model
# compressed_model = engine.compress(model, calibration_dataloader=calib_loader)
```

## References & Theoretical Foundation

- NeuralOps Architectural Specification for `slicegpt_pca`
- Eckart-Young-Mirsky Theorem & Optimal Subspace Approximations

---

*Part of the NeuralOps 72-compressor registry · [Back to Hub](../../index.html)*
<!-- © 2024-2026 Taha Majlesi · MIT License -->
