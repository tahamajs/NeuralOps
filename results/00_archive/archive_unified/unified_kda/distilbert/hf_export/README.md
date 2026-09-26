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
- **Validation Accuracy**: `76.83%`
- **F1 Score (Binary)**: `0.7531`
- **F1 Macro**: `0.7675`
- **Precision**: `0.7752`
- **Recall**: `0.7697`
- **Validation Loss**: `0.6424`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544912`
- **Trainable Recovery Params**: `589902`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `244.17 MB`
- **Throughput**: `18396.52 tokens/sec`
- **Evaluation Time**: `15.26 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
