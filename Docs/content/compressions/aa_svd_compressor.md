# Aasvd

## Overview
True Low-Rank Factorized Linear Layer executing as:
y = (x @ V.T) @ U.T + ((x @ lora_A.T) @ lora_B.T) * lora_scaling + b
Achieving physical parameter reduction from d_in * d_out to r * (d_in + d_out).
Supports optional LoRA adapters for recovery fine-tuning.

Activation-Aware SVD Compressor with streaming EMA covariance accumulation
and true structural module replacement (SVDLinear).

Production hardening features:
- Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
- Safe fallback to original weights by default (fallback_to_original=True).
- Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
- SVD and covariance computations in FP32 with NaN/Inf guards.
- Parameter reduction constraint: r * (in + out) < in * out.
- Comprehensive per-layer stats and error tracking.

## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|
| `in_features` | TODO: Description for in_features |
| `out_features` | TODO: Description for out_features |
| `rank` | TODO: Description for rank |
| `bias` | TODO: Description for bias |
| `use_fused_kernel` | TODO: Description for use_fused_kernel |
| `dtype` | TODO: Description for dtype |
| `device` | TODO: Description for device |
| `lora_rank` | TODO: Description for lora_rank |
| `lora_alpha` | TODO: Description for lora_alpha |
| `lora_r` | TODO: Description for lora_r |
| `freeze_base` | TODO: Description for freeze_base |
| `model` | TODO: Description for model |
| `tokenizer` | TODO: Description for tokenizer |
| `rank` | TODO: Description for rank |
| `ema_beta` | TODO: Description for ema_beta |
| `max_error` | TODO: Description for max_error |
| `fallback_to_original` | TODO: Description for fallback_to_original |
| `calibration_texts` | TODO: Description for calibration_texts |
| `device` | TODO: Description for device |
| `target_modules` | TODO: Description for target_modules |
| `target_attention_only` | TODO: Description for target_attention_only |

## Usage Examples

```python
from Codes.framework.compression_engine import SVDLinear
```
## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | ag_news | `PASS` | 51.00% | **85.80%** | +34.80% | 0.8538 | 1.13x | 16 | N/A | 2026-09-18 |
| **roberta-base** | ag_news | `PASS` | 74.50% | **84.70%** | +10.20% | 0.8406 | 1.08x | 18 | N/A | 2026-09-18 |
| **gpt2** | ag_news | `PASS` | 79.70% | **78.60%** | -1.10% | 0.7824 | 1.03x | 4 | N/A | 2026-09-18 |
| **distilbert-sst2** | mnli | `PASS` | 44.10% | **46.40%** | +2.30% | 0.3658 | 1.13x | 16 | N/A | 2026-09-18 |
| **roberta-base** | mnli | `PASS` | 35.60% | **51.20%** | +15.60% | 0.4609 | 1.08x | 18 | N/A | 2026-09-18 |