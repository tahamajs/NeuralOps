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
- **Validation Accuracy**: `72.71%`
- **F1 Score (Binary)**: `0.7729`
- **F1 Macro**: `0.7155`
- **Precision**: `0.7625`
- **Recall**: `0.7236`
- **Validation Loss**: `0.512`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0596x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `117633974`
- **Trainable Recovery Params**: `2368512`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `374.0 MB`
- **Throughput**: `2810.3 tokens/sec`
- **Evaluation Time**: `191.1 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
