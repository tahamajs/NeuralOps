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
- **Validation Accuracy**: `85.32%`
- **F1 Score (Binary)**: `0.8697`
- **F1 Macro**: `0.8508`
- **Precision**: `0.8714`
- **Recall**: `0.8512`
- **Validation Loss**: `0.6394`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9452x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134707970`
- **Trainable Recovery Params**: `2973696`
- **Compressed / Injected Layers**: `85`
- **Peak VRAM**: `396.58 MB`
- **Throughput**: `10460.39 tokens/sec`
- **Evaluation Time**: `28.48 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
