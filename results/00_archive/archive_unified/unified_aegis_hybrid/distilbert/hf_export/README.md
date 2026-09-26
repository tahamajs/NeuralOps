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
- **Validation Accuracy**: `84.98%`
- **F1 Score (Binary)**: `0.8634`
- **F1 Macro**: `0.8483`
- **Precision**: `0.8599`
- **Recall**: `0.8482`
- **Validation Loss**: `0.3557`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1224x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `59652626`
- **Trainable Recovery Params**: `2134800`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `319.22 MB`
- **Throughput**: `3802.96 tokens/sec`
- **Evaluation Time**: `47.68 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
