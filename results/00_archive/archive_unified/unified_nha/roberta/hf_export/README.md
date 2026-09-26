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
- **Validation Accuracy**: `81.31%`
- **F1 Score (Binary)**: `0.8253`
- **F1 Macro**: `0.8122`
- **Precision**: `0.8166`
- **Recall**: `0.8121`
- **Validation Loss**: `0.3924`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132923150`
- **Trainable Recovery Params**: `1188876`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `663.57 MB`
- **Throughput**: `7908.43 tokens/sec`
- **Evaluation Time**: `53.87 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
