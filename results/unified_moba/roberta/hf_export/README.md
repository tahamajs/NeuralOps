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
- **Validation Accuracy**: `84.86%`
- **F1 Score (Binary)**: `0.8634`
- **F1 Macro**: `0.8468`
- **Precision**: `0.8609`
- **Recall**: `0.8469`
- **Validation Loss**: `0.4284`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9378x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913970`
- **Trainable Recovery Params**: `1771826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `384.31 MB`
- **Throughput**: `37486.46 tokens/sec`
- **Evaluation Time**: `8.08 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
