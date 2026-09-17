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
- **Validation Accuracy**: `59.98%`
- **F1 Score (Binary)**: `0.6161`
- **F1 Macro**: `0.599`
- **Precision**: `0.5996`
- **Recall**: `0.5992`
- **Validation Loss**: `0.6697`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1903x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `104722382`
- **Trainable Recovery Params**: `1336524`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `596.26 MB`
- **Throughput**: `5763.15 tokens/sec`
- **Evaluation Time**: `54.56 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
