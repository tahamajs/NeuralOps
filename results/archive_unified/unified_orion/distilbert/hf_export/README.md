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
- **Validation Accuracy**: `75.23%`
- **F1 Score (Binary)**: `0.7866`
- **F1 Macro**: `0.7457`
- **Precision**: `0.7747`
- **Recall**: `0.7496`
- **Validation Loss**: `0.5628`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.227x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54568706`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `297.51 MB`
- **Throughput**: `16693.56 tokens/sec`
- **Evaluation Time**: `18.22 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
