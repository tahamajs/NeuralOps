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
- **Validation Accuracy**: `90.37%`
- **F1 Score (Binary)**: `0.903`
- **F1 Macro**: `0.9037`
- **Precision**: `0.9044`
- **Recall**: `0.9041`
- **Validation Loss**: `0.251`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0008x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `127222418`
- **Trainable Recovery Params**: `3167378`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `336.26 MB`
- **Throughput**: `11877.02 tokens/sec`
- **Evaluation Time**: `50.24 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
