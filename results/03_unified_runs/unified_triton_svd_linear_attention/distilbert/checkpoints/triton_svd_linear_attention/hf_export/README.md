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

# SVD Linear Attention: distilbert with triton_svd_linear_attention

This repository hosts a validated factorized/compressed checkpoint of **distilbert** using the **triton_svd_linear_attention** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `75.80%`
- **F1 Score**: `0.7827`
- **Precision**: `N/A`
- **Recall**: `N/A`
- **Validation Loss**: `1.0943`
- **Verification Status**: `PASS`
- **Date**: `2026-09-29`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.1523x`
- **Compression Ratio**: `13.21%`
- **FLOPs Reduction**: `0.0%`
- **Factorized Layers**: `6`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `N/A`
- **Peak VRAM**: `359.74 MB`
- **Inference Throughput**: `36157.89 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
