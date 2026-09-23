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
- **Validation Accuracy**: `61.70%`
- **F1 Score (Binary)**: `0.6528`
- **F1 Macro**: `0.6128`
- **Precision**: `0.6195`
- **Recall**: `0.6153`
- **Validation Loss**: `0.6695`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9458x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `131783426`
- **Trainable Recovery Params**: `592130`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `649.47 MB`
- **Throughput**: `11010.78 tokens/sec`
- **Evaluation Time**: `19.09 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
