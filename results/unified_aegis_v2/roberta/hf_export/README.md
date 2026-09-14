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
- **Validation Accuracy**: `72.02%`
- **F1 Score (Binary)**: `0.7505`
- **F1 Macro**: `0.716`
- **Precision**: `0.7297`
- **Recall**: `0.7182`
- **Validation Loss**: `0.5772`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1449x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `108872018`
- **Trainable Recovery Params**: `3090000`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `591.97 MB`
- **Throughput**: `4678.15 tokens/sec`
- **Evaluation Time**: `96.69 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
