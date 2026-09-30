---
tags:
- svd
- model-compression
- linear-attention
- efficiency
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention: gpt2 with triton_cosformer

This repository hosts a validated factorized/compressed checkpoint of **gpt2** using the **triton_cosformer** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `64.45%`
- **F1 Score**: `0.7355`
- **Precision**: `N/A`
- **Recall**: `N/A`
- **Validation Loss**: `0.7166`
- **Verification Status**: `PASS`
- **Date**: `2026-09-29`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.0x`
- **Compression Ratio**: `0.0%`
- **FLOPs Reduction**: `0.0%`
- **Factorized Layers**: `12`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `N/A`
- **Peak VRAM**: `749.37 MB`
- **Inference Throughput**: `18634.41 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
