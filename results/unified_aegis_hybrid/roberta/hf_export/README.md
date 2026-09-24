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
- **Validation Accuracy**: `78.21%`
- **F1 Score (Binary)**: `0.7631`
- **F1 Macro**: `0.7807`
- **Precision**: `0.7931`
- **Recall**: `0.7838`
- **Validation Loss**: `0.4652`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1326x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `110051618`
- **Trainable Recovery Params**: `4269600`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `344.53 MB`
- **Throughput**: `5501.29 tokens/sec`
- **Evaluation Time**: `52.31 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
