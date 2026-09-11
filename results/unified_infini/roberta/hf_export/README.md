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
- **Validation Accuracy**: `56.54%`
- **F1 Score (Binary)**: `0.6855`
- **F1 Macro**: `0.4912`
- **Precision**: `0.6317`
- **Recall**: `0.5585`
- **Validation Loss**: `0.6799`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9296x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `134093642`
- **Trainable Recovery Params**: `2951498`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `406.54 MB`
- **Throughput**: `42634.87 tokens/sec`
- **Evaluation Time**: `7.31 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
