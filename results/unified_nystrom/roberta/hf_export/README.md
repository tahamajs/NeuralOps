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
- **Validation Accuracy**: `81.65%`
- **F1 Score (Binary)**: `0.8261`
- **F1 Macro**: `0.816`
- **Precision**: `0.8184`
- **Recall**: `0.8158`
- **Validation Loss**: `0.4015`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913922`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `651.5 MB`
- **Throughput**: `14203.25 tokens/sec`
- **Evaluation Time**: `20.98 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
