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
- **Validation Accuracy**: `62.84%`
- **F1 Score (Binary)**: `0.7286`
- **F1 Macro**: `0.5698`
- **Precision**: `0.7531`
- **Recall**: `0.6219`
- **Validation Loss**: `1.5005`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1968x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `55945034`
- **Trainable Recovery Params**: `1378634`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `288.41 MB`
- **Throughput**: `14210.76 tokens/sec`
- **Evaluation Time**: `30.61 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
