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
- **Validation Accuracy**: `88.99%`
- **F1 Score (Binary)**: `0.8865`
- **F1 Macro**: `0.8898`
- **Precision**: `0.893`
- **Recall**: `0.8908`
- **Validation Loss**: `0.2882`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544912`
- **Trainable Recovery Params**: `589902`
- **Compressed / Injected Layers**: `6`
- **Peak VRAM**: `374.77 MB`
- **Throughput**: `19188.62 tokens/sec`
- **Evaluation Time**: `14.2 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
