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
- **Validation Accuracy**: `77.18%`
- **F1 Score (Binary)**: `0.7715`
- **F1 Macro**: `0.7718`
- **Precision**: `0.7721`
- **Recall**: `0.7721`
- **Validation Loss**: `0.5183`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.9073x`
- **FLOPs Reduction**: `75.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `35104550`
- **Trainable Recovery Params**: `43814`
- **Compressed / Injected Layers**: `36`
- **Peak VRAM**: `258.7 MB`
- **Throughput**: `155604.92 tokens/sec`
- **Evaluation Time**: `3.98 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
