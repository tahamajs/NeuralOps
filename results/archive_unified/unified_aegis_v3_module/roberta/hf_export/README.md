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
- **Validation Accuracy**: `58.83%`
- **F1 Score (Binary)**: `0.7098`
- **F1 Macro**: `0.5008`
- **Precision**: `0.7452`
- **Recall**: `0.5808`
- **Validation Loss**: `0.6552`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.935x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `133307330`
- **Trainable Recovery Params**: `1573056`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `688.09 MB`
- **Throughput**: `5689.66 tokens/sec`
- **Evaluation Time**: `58.58 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
