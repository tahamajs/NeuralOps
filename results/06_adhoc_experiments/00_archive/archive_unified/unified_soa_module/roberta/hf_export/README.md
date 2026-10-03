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
- **F1 Score (Binary)**: `0.7018`
- **F1 Macro**: `0.5414`
- **Precision**: `0.6702`
- **Recall**: `0.5913`
- **Validation Loss**: `0.769`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0529x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `118389506`
- **Trainable Recovery Params**: `811008`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `595.15 MB`
- **Throughput**: `7661.01 tokens/sec`
- **Evaluation Time**: `41.07 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
