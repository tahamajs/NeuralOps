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
- **Validation Accuracy**: `83.83%`
- **F1 Score (Binary)**: `0.8542`
- **F1 Macro**: `0.8364`
- **Precision**: `0.8504`
- **Recall**: `0.8366`
- **Validation Loss**: `0.3881`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913934`
- **Trainable Recovery Params**: `1179660`
- **Compressed / Injected Layers**: `60`
- **Peak VRAM**: `656.08 MB`
- **Throughput**: `9484.44 tokens/sec`
- **Evaluation Time**: `30.04 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
