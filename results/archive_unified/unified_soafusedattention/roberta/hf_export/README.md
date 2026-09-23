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
- **Validation Accuracy**: `70.41%`
- **F1 Score (Binary)**: `0.7383`
- **F1 Macro**: `0.699`
- **Precision**: `0.7146`
- **Recall**: `0.702`
- **Validation Loss**: `0.5708`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1028x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `113025794`
- **Trainable Recovery Params**: `755712`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `573.39 MB`
- **Throughput**: `7571.78 tokens/sec`
- **Evaluation Time**: `43.38 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
