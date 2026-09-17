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
- **Validation Accuracy**: `59.29%`
- **F1 Score (Binary)**: `0.7097`
- **F1 Macro**: `0.5142`
- **Precision**: `0.7248`
- **Recall**: `0.5857`
- **Validation Loss**: `0.6492`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1641x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `107072450`
- **Trainable Recovery Params**: `1882562`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `325.81 MB`
- **Throughput**: `40505.95 tokens/sec`
- **Evaluation Time**: `14.23 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
