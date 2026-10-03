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
- **Validation Accuracy**: `74.43%`
- **F1 Score (Binary)**: `0.7824`
- **F1 Macro**: `0.7361`
- **Precision**: `0.7712`
- **Recall**: `0.7413`
- **Validation Loss**: `0.5215`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.3806x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `48496398`
- **Trainable Recovery Params**: `514060`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `253.68 MB`
- **Throughput**: `40708.95 tokens/sec`
- **Evaluation Time**: `8.96 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
