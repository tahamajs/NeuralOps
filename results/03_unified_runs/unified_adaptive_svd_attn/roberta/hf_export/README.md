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
- **Validation Accuracy**: `75.11%`
- **F1 Score (Binary)**: `0.7779`
- **F1 Macro**: `0.7475`
- **Precision**: `0.7621`
- **Recall**: `0.7492`
- **Validation Loss**: `0.5155`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.093x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `114039554`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `12`
- **Peak VRAM**: `602.75 MB`
- **Throughput**: `11912.41 tokens/sec`
- **Evaluation Time**: `27.31 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
