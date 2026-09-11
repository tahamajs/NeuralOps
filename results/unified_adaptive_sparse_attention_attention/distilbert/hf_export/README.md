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
- **Validation Accuracy**: `89.91%`
- **F1 Score (Binary)**: `0.9033`
- **F1 Macro**: `0.8989`
- **Precision**: `0.9003`
- **Recall**: `0.8986`
- **Validation Loss**: `0.5722`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8193x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `81719144`
- **Trainable Recovery Params**: `1182056`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `396.52 MB`
- **Throughput**: `16770.04 tokens/sec`
- **Evaluation Time**: `16.28 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
