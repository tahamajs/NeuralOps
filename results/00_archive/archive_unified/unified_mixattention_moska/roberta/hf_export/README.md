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
- **Validation Accuracy**: `86.24%`
- **F1 Score (Binary)**: `0.8704`
- **F1 Macro**: `0.8619`
- **Precision**: `0.8655`
- **Recall**: `0.8615`
- **Validation Loss**: `0.3524`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913970`
- **Trainable Recovery Params**: `1771826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `376.7 MB`
- **Throughput**: `31562.88 tokens/sec`
- **Evaluation Time**: `9.21 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
