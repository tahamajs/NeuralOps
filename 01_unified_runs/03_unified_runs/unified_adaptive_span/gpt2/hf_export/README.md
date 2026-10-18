---
tags:
- svd-linear-attention
- model-compression
- gpt2
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: GPT2 (gpt2)

Validated factorized/adapted checkpoint of **gpt2** using **gpt2** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `77.87%`
- **F1 Score (Binary)**: `0.7689`
- **F1 Macro**: `0.7783`
- **Precision**: `0.7826`
- **Recall**: `0.7797`
- **Validation Loss**: `0.6157`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095696`
- **Trainable Recovery Params**: `2654352`
- **Compressed / Injected Layers**: `12`
- **Peak VRAM**: `441.69 MB`
- **Throughput**: `8085.98 tokens/sec`
- **Evaluation Time**: `35.84 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
