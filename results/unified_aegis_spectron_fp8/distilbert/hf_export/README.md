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
- **Validation Accuracy**: `87.84%`
- **F1 Score (Binary)**: `0.883`
- **F1 Macro**: `0.8783`
- **Precision**: `0.8792`
- **Recall**: `0.878`
- **Validation Loss**: `0.3014`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1177x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `59904860`
- **Trainable Recovery Params**: `1184256`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `230.69 MB`
- **Throughput**: `5243.53 tokens/sec`
- **Evaluation Time**: `102.17 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
