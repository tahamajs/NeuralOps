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
- **Validation Accuracy**: `69.50%`
- **F1 Score (Binary)**: `0.7569`
- **F1 Macro**: `0.6738`
- **Precision**: `0.7509`
- **Recall**: `0.6905`
- **Validation Loss**: `0.5989`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9975x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67121078`
- **Trainable Recovery Params**: `4154174`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `427.0 MB`
- **Throughput**: `9828.24 tokens/sec`
- **Evaluation Time**: `35.7 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
