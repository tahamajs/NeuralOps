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
- **Validation Accuracy**: `87.04%`
- **F1 Score (Binary)**: `0.8732`
- **F1 Macro**: `0.8704`
- **Precision**: `0.8704`
- **Recall**: `0.8703`
- **Validation Loss**: `0.3747`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9909x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67569410`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `196.08 MB`
- **Throughput**: `18433.32 tokens/sec`
- **Evaluation Time**: `15.18 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
