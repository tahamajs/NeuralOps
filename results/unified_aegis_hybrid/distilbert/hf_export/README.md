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
- **F1 Score (Binary)**: `0.805`
- **F1 Macro**: `0.782`
- **Precision**: `0.7935`
- **Recall**: `0.7827`
- **Validation Loss**: `0.6117`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2059x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `55523858`
- **Trainable Recovery Params**: `1237322`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `298.9 MB`
- **Throughput**: `15970.76 tokens/sec`
- **Evaluation Time**: `20.26 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
