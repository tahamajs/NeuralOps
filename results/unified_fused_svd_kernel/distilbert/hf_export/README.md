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
- **Validation Accuracy**: `80.28%`
- **F1 Score (Binary)**: `0.8219`
- **F1 Macro**: `0.8004`
- **Precision**: `0.8131`
- **Recall**: `0.801`
- **Validation Loss**: `0.5295`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2257x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54624074`
- **Trainable Recovery Params**: `1181954`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `280.1 MB`
- **Throughput**: `11815.51 tokens/sec`
- **Evaluation Time**: `20.73 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
