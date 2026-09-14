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
- **Validation Accuracy**: `79.82%`
- **F1 Score (Binary)**: `0.8124`
- **F1 Macro**: `0.797`
- **Precision**: `0.8023`
- **Recall**: `0.797`
- **Validation Loss**: `0.4731`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.093x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `114039554`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `60`
- **Peak VRAM**: `580.81 MB`
- **Throughput**: `13055.63 tokens/sec`
- **Evaluation Time**: `55.08 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
