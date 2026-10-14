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
- **Validation Accuracy**: `80.73%`
- **F1 Score (Binary)**: `0.8209`
- **F1 Macro**: `0.8062`
- **Precision**: `0.8116`
- **Recall**: `0.8062`
- **Validation Loss**: `0.4225`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1932x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `56113682`
- **Trainable Recovery Params**: `2134800`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `212.32 MB`
- **Throughput**: `11005.16 tokens/sec`
- **Evaluation Time**: `27.19 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
