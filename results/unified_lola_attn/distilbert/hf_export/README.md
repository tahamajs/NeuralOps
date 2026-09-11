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
- **Validation Accuracy**: `77.87%`
- **F1 Score (Binary)**: `0.8016`
- **F1 Macro**: `0.7757`
- **Precision**: `0.7899`
- **Recall**: `0.7768`
- **Validation Loss**: `0.6985`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.227x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54568724`
- **Trainable Recovery Params**: `1181972`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `338.53 MB`
- **Throughput**: `77945.82 tokens/sec`
- **Evaluation Time**: `7.87 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
