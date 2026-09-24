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
- **Validation Accuracy**: `78.33%`
- **F1 Score (Binary)**: `0.7966`
- **F1 Macro**: `0.7823`
- **Precision**: `0.7858`
- **Recall**: `0.7823`
- **Validation Loss**: `0.4668`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.227x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54568754`
- **Trainable Recovery Params**: `589872`
- **Compressed / Injected Layers**: `6`
- **Peak VRAM**: `196.89 MB`
- **Throughput**: `12310.4 tokens/sec`
- **Evaluation Time**: `28.79 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
