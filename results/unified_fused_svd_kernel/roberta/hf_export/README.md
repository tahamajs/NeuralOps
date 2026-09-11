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
- **Validation Accuracy**: `70.07%`
- **F1 Score (Binary)**: `0.7439`
- **F1 Macro**: `0.6919`
- **Precision**: `0.7201`
- **Recall**: `0.6978`
- **Validation Loss**: `0.5915`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1641x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `107072450`
- **Trainable Recovery Params**: `1882562`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `560.48 MB`
- **Throughput**: `11835.03 tokens/sec`
- **Evaluation Time**: `53.35 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
