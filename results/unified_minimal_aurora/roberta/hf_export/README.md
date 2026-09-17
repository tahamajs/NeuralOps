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
- **Validation Accuracy**: `64.56%`
- **F1 Score (Binary)**: `0.7193`
- **F1 Macro**: `0.6194`
- **Precision**: `0.6897`
- **Recall**: `0.641`
- **Validation Loss**: `0.665`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.093x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `114039578`
- **Trainable Recovery Params**: `1179672`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `579.52 MB`
- **Throughput**: `13332.46 tokens/sec`
- **Evaluation Time**: `27.4 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
