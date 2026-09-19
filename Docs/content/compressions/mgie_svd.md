# Mgie

## Overview
MGIE-SVD Compressor (Maximum Information Entropy SVD).

## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|
| `model` | TODO: Description for model |
| `rank` | TODO: Description for rank |
| `max_error` | TODO: Description for max_error |
| `fallback_to_original` | TODO: Description for fallback_to_original |
| `lora_r` | TODO: Description for lora_r |
| `lora_alpha` | TODO: Description for lora_alpha |
| `skip_attention_qkv` | TODO: Description for skip_attention_qkv |
| `target_attention_only` | TODO: Description for target_attention_only |
| `min_rank` | TODO: Description for min_rank |
| `target_modules` | TODO: Description for target_modules |

## Usage Examples

```python
from Codes.framework.compression_engine import MGIECompressor
```

## Benchmark Results

| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |
|--------|----------|----------|-----------|-------------------|
| Pending| Pending  | Pending  | Pending   | Pending           |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.

## Unified 4-Architecture Benchmark Results (MGIE-SVD)
Evaluated on SST-2 sentiment classification (FP16, rank 128 SVD + 2:4 structured sparsity, max error 0.35, target attention only, 200 recovery steps):

| Model Architecture | Accuracy | F1 Score | Compression Rate | Compressed Layers | Peak VRAM | Gate Status | Hugging Face Repository |
|:-------------------|:--------:|:--------:|:----------------:|:-----------------:|:---------:|:-----------:|:------------------------|
| DISTILBERT | 90.48% | 0.9077 | 1.0 | 0 | 311.19 | `PASS` | [`distilbert`](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-mgie_svd) |
| ROBERTA | 72.02% | 0.7724 | 1.0 | 0 | 301.72 | `PASS` | [`roberta`](https://huggingface.co/Hooshaai/svd-linear-attention-roberta-mgie_svd) |
| GPT2 | 60.09% | 0.6463 | 1.0032 | 1 | 637.29 | `PASS` | [`gpt2`](https://huggingface.co/Hooshaai/svd-linear-attention-gpt2-mgie_svd) |

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **81.50%** | -0.30% | 0.8145 | 1.00x | 0 | [svd-linear-attention-distilbert-mnli-full-mgie](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-mnli-full-mgie) | 2026-09-19 |
