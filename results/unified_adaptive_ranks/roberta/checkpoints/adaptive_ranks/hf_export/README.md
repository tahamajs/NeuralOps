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

# SVD Linear Attention: roberta with adaptive_ranks

This repository hosts a validated factorized/compressed checkpoint of **roberta** using the **adaptive_ranks** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `79.47%`
- **F1 Score**: `0.8172`
- **Precision**: `0.8085`
- **Recall**: `0.7927`
- **Validation Loss**: `0.9163`
- **Verification Status**: `PASS`
- **Date**: `2026-09-24`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.1862x`
- **Compression Ratio**: `15.7%`
- **FLOPs Reduction**: `69.11%`
- **Factorized Layers**: `48`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `0.7944`
- **Peak VRAM**: `606.24 MB`
- **Inference Throughput**: `20291.88 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
