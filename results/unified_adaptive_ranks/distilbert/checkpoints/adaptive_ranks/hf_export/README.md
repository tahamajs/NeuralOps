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

# SVD Linear Attention: distilbert with adaptive_ranks

This repository hosts a validated factorized/compressed checkpoint of **distilbert** using the **adaptive_ranks** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `82.11%`
- **F1 Score**: `0.8408`
- **Precision**: `0.8367`
- **Recall**: `0.8191`
- **Validation Loss**: `0.4438`
- **Verification Status**: `PASS`
- **Date**: `2026-09-24`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.1739x`
- **Compression Ratio**: `14.81%`
- **FLOPs Reduction**: `70.05%`
- **Factorized Layers**: `24`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `0.7744`
- **Peak VRAM**: `215.81 MB`
- **Inference Throughput**: `38106.43 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
