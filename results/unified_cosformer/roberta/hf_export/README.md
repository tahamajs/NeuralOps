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
- **Validation Accuracy**: `65.48%`
- **F1 Score (Binary)**: `0.731`
- **F1 Macro**: `0.6247`
- **Precision**: `0.7141`
- **Recall**: `0.6498`
- **Validation Loss**: `0.6202`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9452x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134707982`
- **Trainable Recovery Params**: `3565838`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `402.52 MB`
- **Throughput**: `7913.02 tokens/sec`
- **Evaluation Time**: `36.86 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
