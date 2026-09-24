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
- **Validation Accuracy**: `63.19%`
- **F1 Score (Binary)**: `0.7259`
- **F1 Macro**: `0.5828`
- **Precision**: `0.7268`
- **Recall**: `0.6258`
- **Validation Loss**: `0.6208`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2485x`
- **FLOPs Reduction**: `91.67%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `99668832`
- **Trainable Recovery Params**: `1179744`
- **Compressed / Injected Layers**: `12`
- **Peak VRAM**: `667.84 MB`
- **Throughput**: `4787.79 tokens/sec`
- **Evaluation Time**: `81.47 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
