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
- **Validation Accuracy**: `57.45%`
- **F1 Score (Binary)**: `0.6986`
- **F1 Macro**: `0.4877`
- **Precision**: `0.6908`
- **Recall**: `0.5672`
- **Validation Loss**: `0.6916`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913982`
- **Trainable Recovery Params**: `1771838`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `376.24 MB`
- **Throughput**: `34387.94 tokens/sec`
- **Evaluation Time**: `9.16 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
