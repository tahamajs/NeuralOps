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
- **Validation Accuracy**: `78.44%`
- **F1 Score (Binary)**: `0.8029`
- **F1 Macro**: `0.7825`
- **Precision**: `0.7912`
- **Recall**: `0.7829`
- **Validation Loss**: `0.6167`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2059x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `55523882`
- **Trainable Recovery Params**: `2137130`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `208.43 MB`
- **Throughput**: `65663.58 tokens/sec`
- **Evaluation Time**: `7.42 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
