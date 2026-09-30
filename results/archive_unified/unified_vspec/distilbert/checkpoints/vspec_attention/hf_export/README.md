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
- **Validation Accuracy**: `77.52%`
- **F1 Score (Binary)**: `0.7737`
- **F1 Macro**: `0.7752`
- **Precision**: `0.7758`
- **Recall**: `0.7756`
- **Validation Loss**: `0.6088`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1993x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `55828154`
- **Trainable Recovery Params**: `739890`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `280.94 MB`
- **Throughput**: `11633.35 tokens/sec`
- **Evaluation Time**: `24.15 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
