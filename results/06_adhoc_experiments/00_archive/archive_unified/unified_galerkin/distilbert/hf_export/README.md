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
- **Validation Accuracy**: `65.94%`
- **F1 Score (Binary)**: `0.6426`
- **F1 Macro**: `0.6586`
- **Precision**: `0.6625`
- **Recall**: `0.6605`
- **Validation Loss**: `0.7333`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544834`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `322.36 MB`
- **Throughput**: `16067.24 tokens/sec`
- **Evaluation Time**: `16.28 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
