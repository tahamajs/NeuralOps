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
- **Validation Accuracy**: `62.73%`
- **F1 Score (Binary)**: `0.587`
- **F1 Macro**: `0.6237`
- **Precision**: `0.6354`
- **Recall**: `0.6293`
- **Validation Loss**: `0.7828`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913922`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `656.69 MB`
- **Throughput**: `13209.85 tokens/sec`
- **Evaluation Time**: `22.18 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
