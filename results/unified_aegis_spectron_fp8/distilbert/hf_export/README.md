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
- **Validation Accuracy**: `62.39%`
- **F1 Score (Binary)**: `0.7225`
- **F1 Macro**: `0.5694`
- **Precision**: `0.7259`
- **Recall**: `0.6175`
- **Validation Loss**: `0.6748`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2229x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54753068`
- **Trainable Recovery Params**: `760344`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `293.25 MB`
- **Throughput**: `12739.0 tokens/sec`
- **Evaluation Time**: `23.34 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
