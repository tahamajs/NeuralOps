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
- **Validation Accuracy**: `85.44%`
- **F1 Score (Binary)**: `0.867`
- **F1 Macro**: `0.853`
- **Precision**: `0.8635`
- **Recall**: `0.8529`
- **Validation Loss**: `0.5247`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1523x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `58107662`
- **Trainable Recovery Params**: `1181966`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `293.2 MB`
- **Throughput**: `93324.88 tokens/sec`
- **Evaluation Time**: `7.74 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
