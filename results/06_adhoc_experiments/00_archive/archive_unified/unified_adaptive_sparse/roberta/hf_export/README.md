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
- **Validation Accuracy**: `66.51%`
- **F1 Score (Binary)**: `0.6198`
- **F1 Macro**: `0.6603`
- **Precision**: `0.6793`
- **Recall**: `0.6676`
- **Validation Loss**: `0.6724`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8085x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `154175390`
- **Trainable Recovery Params**: `1179660`
- **Compressed / Injected Layers**: `60`
- **Peak VRAM**: `687.46 MB`
- **Throughput**: `9569.17 tokens/sec`
- **Evaluation Time**: `30.96 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
