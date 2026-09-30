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
- **Validation Accuracy**: `81.88%`
- **F1 Score (Binary)**: `0.8316`
- **F1 Macro**: `0.8178`
- **Precision**: `0.8233`
- **Recall**: `0.8177`
- **Validation Loss**: `0.4139`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1516x`
- **FLOPs Reduction**: `57.87%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `108056832`
- **Trainable Recovery Params**: `38400`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `444.13 MB`
- **Throughput**: `24364.3 tokens/sec`
- **Evaluation Time**: `14.75 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
