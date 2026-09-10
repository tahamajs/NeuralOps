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
- **F1 Score (Binary)**: `0.688`
- **F1 Macro**: `0.5605`
- **Precision**: `0.6349`
- **Recall**: `0.5924`
- **Validation Loss**: `0.6655`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9458x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `131783438`
- **Trainable Recovery Params**: `592130`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `363.51 MB`
- **Throughput**: `48620.44 tokens/sec`
- **Evaluation Time**: `4.59 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
