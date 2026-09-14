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
- **Validation Accuracy**: `57.45%`
- **F1 Score (Binary)**: `0.6952`
- **F1 Macro**: `0.4956`
- **Precision**: `0.6675`
- **Recall**: `0.5675`
- **Validation Loss**: `0.6809`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2229x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54753068`
- **Trainable Recovery Params**: `760344`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `287.99 MB`
- **Throughput**: `11786.66 tokens/sec`
- **Evaluation Time**: `25.35 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
