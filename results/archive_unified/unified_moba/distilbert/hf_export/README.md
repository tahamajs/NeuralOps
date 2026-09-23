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
- **Validation Accuracy**: `67.32%`
- **F1 Score (Binary)**: `0.7334`
- **F1 Macro**: `0.6556`
- **Precision**: `0.7083`
- **Recall**: `0.6692`
- **Validation Loss**: `0.6202`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544840`
- **Trainable Recovery Params**: `589830`
- **Compressed / Injected Layers**: `30`
- **Peak VRAM**: `195.26 MB`
- **Throughput**: `19896.61 tokens/sec`
- **Evaluation Time**: `14.03 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
