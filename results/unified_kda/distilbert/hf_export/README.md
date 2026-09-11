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
- **Validation Accuracy**: `76.83%`
- **F1 Score (Binary)**: `0.7584`
- **F1 Macro**: `0.768`
- **Precision**: `0.772`
- **Recall**: `0.7694`
- **Validation Loss**: `0.5225`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544912`
- **Trainable Recovery Params**: `1182032`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `349.3 MB`
- **Throughput**: `18111.8 tokens/sec`
- **Evaluation Time**: `15.83 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
