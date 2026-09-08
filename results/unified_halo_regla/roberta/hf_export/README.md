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
- **Validation Accuracy**: `87.84%`
- **F1 Score (Binary)**: `0.8896`
- **F1 Macro**: `0.8772`
- **Precision**: `0.8899`
- **Recall**: `0.8769`
- **Validation Loss**: `0.4731`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9444x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `134818706`
- **Trainable Recovery Params**: `3565826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `400.47 MB`
- **Throughput**: `39305.88 tokens/sec`
- **Evaluation Time**: `8.76 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
