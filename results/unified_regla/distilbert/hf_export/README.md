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
- **Validation Accuracy**: `74.89%`
- **F1 Score (Binary)**: `0.7497`
- **F1 Macro**: `0.7489`
- **Precision**: `0.749`
- **Recall**: `0.749`
- **Validation Loss**: `0.5117`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9397x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `71254274`
- **Trainable Recovery Params**: `744194`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `344.72 MB`
- **Throughput**: `63826.21 tokens/sec`
- **Evaluation Time**: `5.61 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
