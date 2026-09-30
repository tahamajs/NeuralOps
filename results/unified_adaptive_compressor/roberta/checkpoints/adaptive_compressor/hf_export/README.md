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
- **Validation Accuracy**: `86.35%`
- **F1 Score (Binary)**: `0.8649`
- **F1 Macro**: `0.8635`
- **Precision**: `0.8635`
- **Recall**: `0.8636`
- **Validation Loss**: `0.3554`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.121x`
- **FLOPs Reduction**: `47.53%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `111191810`
- **Trainable Recovery Params**: `628994`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `580.02 MB`
- **Throughput**: `17948.91 tokens/sec`
- **Evaluation Time**: `25.58 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
