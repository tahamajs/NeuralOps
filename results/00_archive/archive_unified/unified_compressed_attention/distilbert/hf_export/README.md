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
- **Validation Accuracy**: `74.54%`
- **F1 Score (Binary)**: `0.7711`
- **F1 Macro**: `0.7422`
- **Precision**: `0.7544`
- **Recall**: `0.7436`
- **Validation Loss**: `0.5396`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.237x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54126728`
- **Trainable Recovery Params**: `1345926`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `283.86 MB`
- **Throughput**: `30316.81 tokens/sec`
- **Evaluation Time**: `11.55 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
