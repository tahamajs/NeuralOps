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
- **Validation Accuracy**: `74.77%`
- **F1 Score (Binary)**: `0.7727`
- **F1 Macro**: `0.7446`
- **Precision**: `0.7563`
- **Recall**: `0.7459`
- **Validation Loss**: `0.5126`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9411x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `135297986`
- **Trainable Recovery Params**: `3563712`
- **Compressed / Injected Layers**: `85`
- **Peak VRAM**: `410.26 MB`
- **Throughput**: `6820.9 tokens/sec`
- **Evaluation Time**: `64.53 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
