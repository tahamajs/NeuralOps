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
- **Validation Accuracy**: `75.46%`
- **F1 Score (Binary)**: `0.7068`
- **F1 Macro**: `0.7479`
- **Precision**: `0.7923`
- **Recall**: `0.7578`
- **Validation Loss**: `0.8643`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.7493x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `89359142`
- **Trainable Recovery Params**: `1771814`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `2323.24 MB`
- **Throughput**: `4342.97 tokens/sec`
- **Evaluation Time**: `33.92 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
