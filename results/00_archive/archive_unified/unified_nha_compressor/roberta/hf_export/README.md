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
- **Validation Accuracy**: `81.54%`
- **F1 Score (Binary)**: `0.8267`
- **F1 Macro**: `0.8146`
- **Precision**: `0.8184`
- **Recall**: `0.8144`
- **Validation Loss**: `0.389`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132923150`
- **Trainable Recovery Params**: `1188876`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `664.71 MB`
- **Throughput**: `7219.91 tokens/sec`
- **Evaluation Time**: `57.59 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
