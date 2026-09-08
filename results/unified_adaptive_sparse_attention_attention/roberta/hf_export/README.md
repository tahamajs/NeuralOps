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
- **Validation Accuracy**: `83.83%`
- **F1 Score (Binary)**: `0.8589`
- **F1 Macro**: `0.8348`
- **Precision**: `0.8628`
- **Recall**: `0.8359`
- **Validation Loss**: `0.4959`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.7767x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `163941266`
- **Trainable Recovery Params**: `4450706`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `420.6 MB`
- **Throughput**: `40879.57 tokens/sec`
- **Evaluation Time**: `7.87 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
