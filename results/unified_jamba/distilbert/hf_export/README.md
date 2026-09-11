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
- **Validation Accuracy**: `87.16%`
- **F1 Score (Binary)**: `0.8756`
- **F1 Macro**: `0.8714`
- **Precision**: `0.8719`
- **Recall**: `0.8713`
- **Validation Loss**: `0.5107`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.4176x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `160322336`
- **Trainable Recovery Params**: `1698080`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `695.65 MB`
- **Throughput**: `2396.61 tokens/sec`
- **Evaluation Time**: `158.18 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
