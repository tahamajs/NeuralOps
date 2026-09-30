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
- **Validation Accuracy**: `90.83%`
- **F1 Score (Binary)**: `0.9113`
- **F1 Macro**: `0.9081`
- **Precision**: `0.9088`
- **Recall**: `0.9079`
- **Validation Loss**: `0.3585`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `66955010`
- **Trainable Recovery Params**: `6914`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `308.34 MB`
- **Throughput**: `143115.09 tokens/sec`
- **Evaluation Time**: `4.07 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
