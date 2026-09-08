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
- **Validation Accuracy**: `65.71%`
- **F1 Score (Binary)**: `0.7337`
- **F1 Macro**: `0.6261`
- **Precision**: `0.7205`
- **Recall**: `0.652`
- **Validation Loss**: `0.6107`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9411x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `135297938`
- **Trainable Recovery Params**: `4155650`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `407.41 MB`
- **Throughput**: `6943.57 tokens/sec`
- **Evaluation Time**: `39.26 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
