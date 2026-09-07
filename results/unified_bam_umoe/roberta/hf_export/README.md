---
tags:
- svd-linear-attention
- model-compression
- roberta
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: ROBERTA (roberta)

Validated factorized/adapted checkpoint of **roberta** using **roberta** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `84.63%`
- **F1 Score (Binary)**: `0.863`
- **F1 Macro**: `0.844`
- **Precision**: `0.8626`
- **Recall**: `0.8444`
- **Validation Loss**: `0.4533`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.3522x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `361513742`
- **Trainable Recovery Params**: `3565838`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `849.7 MB`
- **Throughput**: `6897.2 tokens/sec`
- **Evaluation Time**: `45.47 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
