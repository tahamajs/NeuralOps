---
tags:
- svd-linear-attention
- model-compression
- distilbert
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: DISTILBERT (distilbert)

Validated factorized/adapted checkpoint of **distilbert** using **distilbert** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `86.47%`
- **F1 Score (Binary)**: `0.8628`
- **F1 Macro**: `0.8647`
- **Precision**: `0.8659`
- **Recall**: `0.8652`
- **Validation Loss**: `0.4056`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9996x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `66979592`
- **Trainable Recovery Params**: `592130`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `321.79 MB`
- **Throughput**: `68267.54 tokens/sec`
- **Evaluation Time**: `3.47 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
