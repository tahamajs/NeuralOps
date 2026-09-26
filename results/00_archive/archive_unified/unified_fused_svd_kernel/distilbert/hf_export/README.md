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
- **Validation Accuracy**: `79.93%`
- **F1 Score (Binary)**: `0.8201`
- **F1 Macro**: `0.7966`
- **Precision**: `0.8115`
- **Recall**: `0.7975`
- **Validation Loss**: `0.4844`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2404x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `53978882`
- **Trainable Recovery Params**: `1184256`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `280.27 MB`
- **Throughput**: `3888.43 tokens/sec`
- **Evaluation Time**: `49.95 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
