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
- **F1 Score (Binary)**: `0.623`
- **F1 Macro**: `0.5956`
- **Precision**: `0.5977`
- **Recall**: `0.5964`
- **Validation Loss**: `0.6689`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.706x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `176542538`
- **Trainable Recovery Params**: `2361674`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `2738.19 MB`
- **Throughput**: `2203.51 tokens/sec`
- **Evaluation Time**: `105.01 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
