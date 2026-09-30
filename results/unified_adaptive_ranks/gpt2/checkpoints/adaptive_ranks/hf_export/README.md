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

# SVD Linear Attention: gpt2 with adaptive_ranks

This repository hosts a validated factorized/compressed checkpoint of **gpt2** using the **adaptive_ranks** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `79.82%`
- **F1 Score**: `0.8182`
- **Precision**: `0.8089`
- **Recall**: `0.7964`
- **Validation Loss**: `0.4495`
- **Verification Status**: `PASS`
- **Date**: `2026-09-24`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.166x`
- **Compression Ratio**: `14.23%`
- **FLOPs Reduction**: `62.57%`
- **Factorized Layers**: `24`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `0.7116`
- **Peak VRAM**: `441.6 MB`
- **Inference Throughput**: `24047.11 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
