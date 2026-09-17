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
- **Validation Accuracy**: `63.65%`
- **F1 Score (Binary)**: `0.7325`
- **F1 Macro**: `0.5827`
- **Precision**: `0.7547`
- **Recall**: `0.6301`
- **Validation Loss**: `0.5739`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.3092x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `216526142`
- **Trainable Recovery Params**: `1181984`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1604.75 MB`
- **Throughput**: `8212.62 tokens/sec`
- **Evaluation Time**: `40.46 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
