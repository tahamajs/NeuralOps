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
- **Validation Accuracy**: `64.79%`
- **F1 Score (Binary)**: `0.5846`
- **F1 Macro**: `0.6396`
- **Precision**: `0.6685`
- **Recall**: `0.651`
- **Validation Loss**: `0.7273`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.7729x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `161262494`
- **Trainable Recovery Params**: `1179804`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `689.25 MB`
- **Throughput**: `8460.41 tokens/sec`
- **Evaluation Time**: `36.64 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
