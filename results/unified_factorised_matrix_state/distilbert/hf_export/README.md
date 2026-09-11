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
- **Validation Accuracy**: `77.06%`
- **F1 Score (Binary)**: `0.7608`
- **F1 Macro**: `0.7703`
- **Precision**: `0.7744`
- **Recall**: `0.7717`
- **Validation Loss**: `0.5656`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.987x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67839842`
- **Trainable Recovery Params**: `1476962`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `342.49 MB`
- **Throughput**: `53181.74 tokens/sec`
- **Evaluation Time**: `5.44 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
