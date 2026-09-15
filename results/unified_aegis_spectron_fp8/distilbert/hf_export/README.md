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
- **Validation Accuracy**: `79.59%`
- **F1 Score (Binary)**: `0.8118`
- **F1 Macro**: `0.7944`
- **Precision**: `0.8014`
- **Recall**: `0.7946`
- **Validation Loss**: `0.4587`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2229x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54753068`
- **Trainable Recovery Params**: `760344`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `290.29 MB`
- **Throughput**: `9858.72 tokens/sec`
- **Evaluation Time**: `28.87 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
