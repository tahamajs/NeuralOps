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
- **Validation Accuracy**: `67.89%`
- **F1 Score (Binary)**: `0.6602`
- **F1 Macro**: `0.6779`
- **Precision**: `0.6831`
- **Recall**: `0.6801`
- **Validation Loss**: `0.6049`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9373x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132990758`
- **Trainable Recovery Params**: `1207332`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `652.3 MB`
- **Throughput**: `13696.59 tokens/sec`
- **Evaluation Time**: `20.9 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
