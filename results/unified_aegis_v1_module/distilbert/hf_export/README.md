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
- **Validation Accuracy**: `80.16%`
- **F1 Score (Binary)**: `0.813`
- **F1 Macro**: `0.8009`
- **Precision**: `0.8039`
- **Recall**: `0.8008`
- **Validation Loss**: `0.5133`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1968x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `55945034`
- **Trainable Recovery Params**: `1378634`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `279.52 MB`
- **Throughput**: `14569.17 tokens/sec`
- **Evaluation Time**: `30.15 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
