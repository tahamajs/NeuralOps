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
- **Accuracy**: `76.95%`
- **F1 Score**: `0.8016`
- **Precision**: `0.7941`
- **Recall**: `0.7668`
- **Validation Loss**: `0.4773`
- **Verification Status**: `PASS`
- **Date**: `2026-09-15`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.0021x`
- **Compression Ratio**: `0.21%`
- **FLOPs Reduction**: `44.79%`
- **Factorized Layers**: `1`
- **Fallback Layers**: `23`
- **Avg Reconstruction Error**: `0.1982078150637553`
- **Peak VRAM**: `638.89 MB`
- **Inference Throughput**: `23688.0 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
