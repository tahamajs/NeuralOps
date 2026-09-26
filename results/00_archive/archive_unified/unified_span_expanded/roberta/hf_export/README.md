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
- **Validation Accuracy**: `75.57%`
- **F1 Score (Binary)**: `0.7966`
- **F1 Macro**: `0.7455`
- **Precision**: `0.7956`
- **Recall**: `0.7523`
- **Validation Loss**: `0.5342`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913970`
- **Trainable Recovery Params**: `1771826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `378.78 MB`
- **Throughput**: `39412.9 tokens/sec`
- **Evaluation Time**: `8.16 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
