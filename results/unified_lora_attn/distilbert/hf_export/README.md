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
- **Validation Accuracy**: `90.60%`
- **F1 Score (Binary)**: `0.9091`
- **F1 Macro**: `0.9059`
- **Precision**: `0.9065`
- **Recall**: `0.9056`
- **Validation Loss**: `0.2598`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9827x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `68134682`
- **Trainable Recovery Params**: `1179672`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `204.58 MB`
- **Throughput**: `26984.05 tokens/sec`
- **Evaluation Time**: `10.77 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
