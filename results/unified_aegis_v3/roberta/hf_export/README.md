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
- **Validation Accuracy**: `86.47%`
- **F1 Score (Binary)**: `0.8706`
- **F1 Macro**: `0.8644`
- **Precision**: `0.866`
- **Recall**: `0.8641`
- **Validation Loss**: `0.3717`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9424x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `135101378`
- **Trainable Recovery Params**: `3367104`
- **Compressed / Injected Layers**: `85`
- **Peak VRAM**: `404.47 MB`
- **Throughput**: `5921.14 tokens/sec`
- **Evaluation Time**: `82.53 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
