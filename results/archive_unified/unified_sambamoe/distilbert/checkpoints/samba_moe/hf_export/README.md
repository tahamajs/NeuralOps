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
- **Validation Accuracy**: `76.72%`
- **F1 Score (Binary)**: `0.8012`
- **F1 Macro**: `0.7602`
- **Precision**: `0.7951`
- **Recall**: `0.7643`
- **Validation Loss**: `0.4916`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.5357x`
- **FLOPs Reduction**: `62.5%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `124983590`
- **Trainable Recovery Params**: `1716518`
- **Compressed / Injected Layers**: `96`
- **Peak VRAM**: `600.35 MB`
- **Throughput**: `18363.68 tokens/sec`
- **Evaluation Time**: `16.73 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
