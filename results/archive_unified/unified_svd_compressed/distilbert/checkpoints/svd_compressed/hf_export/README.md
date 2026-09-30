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
- **Validation Accuracy**: `87.39%`
- **F1 Score (Binary)**: `0.8825`
- **F1 Macro**: `0.8732`
- **Precision**: `0.8789`
- **Recall**: `0.8728`
- **Validation Loss**: `0.3454`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1523x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `58107650`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `283.97 MB`
- **Throughput**: `5973.8 tokens/sec`
- **Evaluation Time**: `303.81 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
