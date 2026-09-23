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
- **Validation Accuracy**: `62.73%`
- **F1 Score (Binary)**: `0.7043`
- **F1 Macro**: `0.6002`
- **Precision**: `0.6641`
- **Recall**: `0.6227`
- **Validation Loss**: `0.6615`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1641x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `107072402`
- **Trainable Recovery Params**: `1290384`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `555.07 MB`
- **Throughput**: `10877.86 tokens/sec`
- **Evaluation Time**: `29.08 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
