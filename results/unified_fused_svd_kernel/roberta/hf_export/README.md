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
- **Validation Accuracy**: `75.92%`
- **F1 Score (Binary)**: `0.7294`
- **F1 Macro**: `0.7562`
- **Precision**: `0.7771`
- **Recall**: `0.7615`
- **Validation Loss**: `0.5308`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1696x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `108866450`
- **Trainable Recovery Params**: `3565826`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `346.5 MB`
- **Throughput**: `1914.33 tokens/sec`
- **Evaluation Time**: `96.88 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
