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
- **Validation Accuracy**: `80.39%`
- **F1 Score (Binary)**: `0.8338`
- **F1 Macro**: `0.7973`
- **Precision**: `0.8405`
- **Recall**: `0.8009`
- **Validation Loss**: `0.6697`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9884x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67741514`
- **Trainable Recovery Params**: `786504`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `203.26 MB`
- **Throughput**: `13898.96 tokens/sec`
- **Evaluation Time**: `25.95 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
