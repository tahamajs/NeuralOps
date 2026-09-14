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
- **Validation Accuracy**: `59.75%`
- **F1 Score (Binary)**: `0.6648`
- **F1 Macro**: `0.5806`
- **Precision**: `0.6101`
- **Recall**: `0.594`
- **Validation Loss**: `0.6836`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1903x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `104722382`
- **Trainable Recovery Params**: `1336524`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `595.4 MB`
- **Throughput**: `7102.0 tokens/sec`
- **Evaluation Time**: `44.9 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
