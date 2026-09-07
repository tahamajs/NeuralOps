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
- **Validation Accuracy**: `83.60%`
- **F1 Score (Binary)**: `0.8566`
- **F1 Macro**: `0.8326`
- **Precision**: `0.8594`
- **Recall**: `0.8337`
- **Validation Loss**: `0.4531`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.937x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `135887630`
- **Trainable Recovery Params**: `3565826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `405.56 MB`
- **Throughput**: `9728.68 tokens/sec`
- **Evaluation Time**: `30.97 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
