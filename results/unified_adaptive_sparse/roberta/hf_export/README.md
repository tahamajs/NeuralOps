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
- **Validation Accuracy**: `58.14%`
- **F1 Score (Binary)**: `0.6883`
- **F1 Macro**: `0.5257`
- **Precision**: `0.6358`
- **Recall**: `0.5753`
- **Validation Loss**: `0.6961`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.7729x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `161262542`
- **Trainable Recovery Params**: `1771982`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `702.77 MB`
- **Throughput**: `11163.83 tokens/sec`
- **Evaluation Time**: `25.77 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
