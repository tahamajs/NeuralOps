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
- **Validation Accuracy**: `89.11%`
- **F1 Score (Binary)**: `0.8917`
- **F1 Macro**: `0.8911`
- **Precision**: `0.8911`
- **Recall**: `0.8912`
- **Validation Loss**: `0.4836`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9907x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67583276`
- **Trainable Recovery Params**: `1220396`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `339.63 MB`
- **Throughput**: `22259.98 tokens/sec`
- **Evaluation Time**: `12.5 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
