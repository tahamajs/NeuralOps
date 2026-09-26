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
- **Validation Accuracy**: `90.25%`
- **F1 Score (Binary)**: `0.9069`
- **F1 Macro**: `0.9023`
- **Precision**: `0.9041`
- **Recall**: `0.902`
- **Validation Loss**: `0.5881`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9827x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `68134694`
- **Trainable Recovery Params**: `1771814`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `352.72 MB`
- **Throughput**: `86872.28 tokens/sec`
- **Evaluation Time**: `3.99 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
