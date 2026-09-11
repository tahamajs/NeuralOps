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
- **Validation Accuracy**: `59.52%`
- **F1 Score (Binary)**: `0.6933`
- **F1 Macro**: `0.549`
- **Precision**: `0.6458`
- **Recall**: `0.5895`
- **Validation Loss**: `0.6883`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9296x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `134093618`
- **Trainable Recovery Params**: `2951474`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `670.9 MB`
- **Throughput**: `47772.09 tokens/sec`
- **Evaluation Time**: `6.5 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
