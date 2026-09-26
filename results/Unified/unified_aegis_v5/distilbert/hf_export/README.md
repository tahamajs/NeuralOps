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
- **Validation Accuracy**: `86.01%`
- **F1 Score (Binary)**: `0.8644`
- **F1 Macro**: `0.8599`
- **Precision**: `0.8604`
- **Recall**: `0.8598`
- **Validation Loss**: `0.3713`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1073x`
- **FLOPs Reduction**: `50.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `60466994`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `203.64 MB`
- **Throughput**: `12628.35 tokens/sec`
- **Evaluation Time**: `26.52 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
