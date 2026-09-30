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

# SVD Linear Attention: gpt2 with triton_svd_linear_attention

This repository hosts a validated factorized/compressed checkpoint of **gpt2** using the **triton_svd_linear_attention** module evaluated on **sst2**.

## Evaluation Summary
- **Accuracy**: `63.99%`
- **F1 Score**: `0.7312`
- **Precision**: `N/A`
- **Recall**: `N/A`
- **Validation Loss**: `0.6321`
- **Verification Status**: `PASS`
- **Date**: `2026-09-29`

## Efficiency & Compression Profile
- **Original Parameters**: `N/A`
- **Compressed Parameters**: `N/A`
- **Compression Rate**: `1.1658x`
- **Compression Ratio**: `14.22%`
- **FLOPs Reduction**: `0.0%`
- **Factorized Layers**: `12`
- **Fallback Layers**: `0`
- **Avg Reconstruction Error**: `N/A`
- **Peak VRAM**: `388.75 MB`
- **Inference Throughput**: `17564.92 tokens/sec`

## Benchmark Configuration
```json
{}
```

## Reproducibility
Trained and factorized via the Unified SVD Linear Attention Framework.
