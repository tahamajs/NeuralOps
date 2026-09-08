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
- **Validation Accuracy**: `85.21%`
- **F1 Score (Binary)**: `0.8584`
- **F1 Macro**: `0.8518`
- **Precision**: `0.8532`
- **Recall**: `0.8515`
- **Validation Loss**: `0.4799`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9447x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134784806`
- **Trainable Recovery Params**: `3614978`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `401.12 MB`
- **Throughput**: `3745.94 tokens/sec`
- **Evaluation Time**: `73.32 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
