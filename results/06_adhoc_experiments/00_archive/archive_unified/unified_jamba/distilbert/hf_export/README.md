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
- **F1 Score (Binary)**: `0.8753`
- **F1 Macro**: `0.8714`
- **Precision**: `0.8718`
- **Recall**: `0.8713`
- **Validation Loss**: `0.5137`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.4176x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `160322336`
- **Trainable Recovery Params**: `1698080`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `716.63 MB`
- **Throughput**: `25763.56 tokens/sec`
- **Evaluation Time**: `12.4 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
