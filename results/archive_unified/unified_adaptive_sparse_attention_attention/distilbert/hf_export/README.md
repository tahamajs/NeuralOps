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
- **Validation Accuracy**: `89.11%`
- **F1 Score (Binary)**: `0.8878`
- **F1 Macro**: `0.891`
- **Precision**: `0.894`
- **Recall**: `0.8919`
- **Validation Loss**: `0.2881`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8193x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `81719120`
- **Trainable Recovery Params**: `589902`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `230.5 MB`
- **Throughput**: `18163.57 tokens/sec`
- **Evaluation Time**: `15.36 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
