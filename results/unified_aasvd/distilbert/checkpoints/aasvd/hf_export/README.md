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

# SVD Linear Attention: distilbert with aasvd

This repository hosts a validated factorized/compressed checkpoint of **distilbert** using the **aasvd** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `84.06%`
- **F1 Score**: `0.8526`
- **Precision**: `0.8465`
- **Recall**: `0.8394`
- **Validation Loss**: `0.5732`
- **Verification Status**: `PASS`
- **Date**: `2026-09-24`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.2042x`
- **Compression Ratio**: `16.96%`
- **FLOPs Reduction**: `80.21%`
- **Factorized Layers**: `24`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `0.8162`
- **Peak VRAM**: `202.06 MB`
- **Inference Throughput**: `41391.05 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
