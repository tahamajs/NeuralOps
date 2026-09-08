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
- **Validation Accuracy**: `60.21%`
- **F1 Score (Binary)**: `0.7186`
- **F1 Macro**: `0.5198`
- **Precision**: `0.7747`
- **Recall**: `0.5947`
- **Validation Loss**: `0.6496`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.2943x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `432670514`
- **Trainable Recovery Params**: `3879170`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1686.96 MB`
- **Throughput**: `6427.6 tokens/sec`
- **Evaluation Time**: `48.59 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
