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
- **Validation Accuracy**: `88.19%`
- **F1 Score (Binary)**: `0.8874`
- **F1 Macro**: `0.8816`
- **Precision**: `0.8836`
- **Recall**: `0.8813`
- **Validation Loss**: `0.8815`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9401x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `71217518`
- **Trainable Recovery Params**: `1329518`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `369.27 MB`
- **Throughput**: `16635.84 tokens/sec`
- **Evaluation Time**: `17.44 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
