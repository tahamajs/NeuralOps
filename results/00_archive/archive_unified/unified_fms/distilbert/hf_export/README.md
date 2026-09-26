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
- **Validation Accuracy**: `76.15%`
- **F1 Score (Binary)**: `0.7432`
- **F1 Macro**: `0.7603`
- **Precision**: `0.7699`
- **Recall**: `0.763`
- **Validation Loss**: `0.6873`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.987x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67839842`
- **Trainable Recovery Params**: `884832`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `339.99 MB`
- **Throughput**: `15783.59 tokens/sec`
- **Evaluation Time**: `17.79 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
