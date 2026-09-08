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
- **Validation Accuracy**: `90.14%`
- **F1 Score (Binary)**: `0.9059`
- **F1 Macro**: `0.9011`
- **Precision**: `0.9031`
- **Recall**: `0.9008`
- **Validation Loss**: `0.5786`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8193x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `81719114`
- **Trainable Recovery Params**: `1182026`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `238.51 MB`
- **Throughput**: `110067.11 tokens/sec`
- **Evaluation Time**: `3.01 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
