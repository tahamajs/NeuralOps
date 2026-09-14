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
- **Validation Accuracy**: `85.32%`
- **F1 Score (Binary)**: `0.8388`
- **F1 Macro**: `0.852`
- **Precision**: `0.8694`
- **Recall**: `0.8551`
- **Validation Loss**: `0.36`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.7767x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `163941278`
- **Trainable Recovery Params**: `3858588`
- **Compressed / Injected Layers**: `121`
- **Peak VRAM**: `411.87 MB`
- **Throughput**: `7499.47 tokens/sec`
- **Evaluation Time**: `39.11 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
