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
- **Validation Accuracy**: `90.94%`
- **F1 Score (Binary)**: `0.9135`
- **F1 Macro**: `0.9092`
- **Precision**: `0.9111`
- **Recall**: `0.9088`
- **Validation Loss**: `0.3499`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9452x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134707970`
- **Trainable Recovery Params**: `3565826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `400.09 MB`
- **Throughput**: `32287.17 tokens/sec`
- **Evaluation Time**: `8.79 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
