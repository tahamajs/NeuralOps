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

# SVD Linear Attention: distilbert with triton_cosformer

This repository hosts a validated factorized/compressed checkpoint of **distilbert** using the **triton_cosformer** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `77.29%`
- **F1 Score**: `0.7916`
- **Precision**: `N/A`
- **Recall**: `N/A`
- **Validation Loss**: `0.7732`
- **Verification Status**: `PASS`
- **Date**: `2026-09-29`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.0x`
- **Compression Ratio**: `0.0%`
- **FLOPs Reduction**: `0.0%`
- **Factorized Layers**: `6`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `N/A`
- **Peak VRAM**: `422.74 MB`
- **Inference Throughput**: `40896.87 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
