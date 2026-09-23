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
- **Validation Accuracy**: `80.16%`
- **F1 Score (Binary)**: `0.8211`
- **F1 Macro**: `0.7992`
- **Precision**: `0.8122`
- **Recall**: `0.7999`
- **Validation Loss**: `0.5334`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2257x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54624098`
- **Trainable Recovery Params**: `1237346`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `280.88 MB`
- **Throughput**: `60886.12 tokens/sec`
- **Evaluation Time**: `8.52 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
