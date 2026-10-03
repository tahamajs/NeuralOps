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
- **Validation Accuracy**: `63.76%`
- **F1 Score (Binary)**: `0.6708`
- **F1 Macro**: `0.6339`
- **Precision**: `0.6407`
- **Recall**: `0.636`
- **Validation Loss**: `0.7007`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9902x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67618682`
- **Trainable Recovery Params**: `1255802`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `334.35 MB`
- **Throughput**: `14773.2 tokens/sec`
- **Evaluation Time**: `16.49 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
