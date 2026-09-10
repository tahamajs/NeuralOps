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
- **Validation Accuracy**: `68.23%`
- **F1 Score (Binary)**: `0.748`
- **F1 Macro**: `0.6592`
- **Precision**: `0.7377`
- **Recall**: `0.6778`
- **Validation Loss**: `0.7814`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.3806x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `48496398`
- **Trainable Recovery Params**: `4649730`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `188.93 MB`
- **Throughput**: `141357.56 tokens/sec`
- **Evaluation Time**: `3.99 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
