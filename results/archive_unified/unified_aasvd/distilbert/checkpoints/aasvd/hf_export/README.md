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
- **Accuracy**: `88.42%`
- **F1 Score**: `0.8889`
- **Precision**: `0.8853`
- **Recall**: `0.8837`
- **Validation Loss**: `0.4048`
- **Verification Status**: `PASS`
- **Date**: `2026-09-21`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.1331x`
- **Compression Ratio**: `11.75%`
- **FLOPs Reduction**: `83.33%`
- **Factorized Layers**: `16`
- **Fallback Layers**: `8`
- **Avg Reconstruction Error**: `0.8352`
- **Peak VRAM**: `317.79 MB`
- **Inference Throughput**: `44102.04 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
