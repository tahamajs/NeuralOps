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
- **Validation Accuracy**: `64.56%`
- **F1 Score (Binary)**: `0.7357`
- **F1 Macro**: `0.5991`
- **Precision**: `0.7489`
- **Recall**: `0.6396`
- **Validation Loss**: `0.5878`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9447x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134784806`
- **Trainable Recovery Params**: `2976804`
- **Compressed / Injected Layers**: `85`
- **Peak VRAM**: `395.46 MB`
- **Throughput**: `11162.46 tokens/sec`
- **Evaluation Time**: `26.51 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
