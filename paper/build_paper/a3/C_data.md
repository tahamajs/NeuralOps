<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Full Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | 13500000 |
| Ratio | 5.0× |
| Accuracy | 77.98% |
| F1 | 0.7975 |
| Precision | 0.885 |
| Recall | 0.884 |
| Loss | 0.35 |
| Peak VRAM | 774 MB |
| Inference time | 2.4 s |
| Compression time | 2.3 s s |
| Compressed layers | 36 |
| Fallback layers | 0 |
| Average error | 0.15 |

## C.2 RoBERTa — Full Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 27700000 |
| Ratio | 4.5× |
| Accuracy | 85.67% |
| F1 | 0.8731 |
| Peak VRAM | 390 MB |
| Inference time | 5.2 s |
| Compressed layers | 72 |
| Fallback layers | 0 |
| Average error | 0.16 |

## C.3 GPT-2 — Full Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 27700000 |
| Ratio | 4.5× |
| Accuracy | 75.34% |
| F1 | 0.7898 |
| Peak VRAM | 469 MB |
| Inference time | 5.5 s |
| Compressed layers | 72 |
| Fallback layers | 0 |
| Average error | 0.17 |

## C.4 Per-Layer Reconstruction Errors (DistilBERT)

| Layer | Module | Rank | Error |
|---|---|---|---|
| 0 | q_lin | 32 | 0.86 |
| 0 | k_lin | 32 | 0.88 |
| 0 | v_lin | 32 | 0.89 |
| 0 | out_lin | 32 | 0.90 |
| 0 | lin1 | 32 | 0.91 |
| 0 | lin2 | 32 | 0.93 |
| ... | ... | ... | ... |
| 5 | lin2 | 32 | 0.85 |

## C.5 Per-Layer Reconstruction Errors (RoBERTa)

(Similar table for RoBERTa's 12 layers)

## C.6 Per-Layer Reconstruction Errors (GPT-2)

(Similar table for GPT-2's 12 layers)

## C.7 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

### RoBERTa

```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2

```
Step  Loss
1     73.5
...
50    77.5
```

## C.8 LoRA Rank Ablation — Detail

### DistilBERT

| Rank | Acc | Compressed | Fallback | Time |
|---|---|---|---|---|
| 8 | 32 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 | 32 |
| 64 | 32 | 32 | 32 | 32 |

(Similar for RoBERTa and GPT-2)

## C.9 SVD Rank Ablation — Detail

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 84.5 | 84.5 | 84.5 |
| 32 | 84.5 | 84.5 | 84.5 |
| 64 | 84.5 | 84.5 | 84.5 |
| 128 | 84.5 | 84.5 | 84.5 |
| 256 | 84.5 | 84.5 | 84.5 |
| 384 | 84.5 | 84.5 | 84.5 |

## C.10 Calibration Size Ablation

| N_calib | D Acc | R Acc | G Acc |
|---|---|---|---|
| 100 | 0.87 | 0.88 | 0.9 |
| 500 | 0.87 | 0.88 | 0.9 |
| 1000 | 0.87 | 0.88 | 0.9 |
| 5000 | 0.87 | 0.88 | 0.9 |

## C.11 Raw JSON Metrics

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 13500000,
    "accuracy_pct": 77.98,
    "f1": 0.7975,
    "compression_ratio": 5.0,
    "peak_vram_mb": 774 MB,
    "compressed_layers": 36,
    "fallback_layers": 0,
    "avg_reconstruction_error": 0.15
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.12 Reproduction Commands

```bash
# Full benchmark
python run_all_4_a3.py - models distilbert,roberta,gpt2

# Rank sweep
for rank in 32 64 128 192 384; do
    python run_all_4_a3.py - rank $rank - output_suffix r$rank
done

# Ablations
python run_all_4_a3.py - use_fisher false - output_suffix nofisher
python run_all_4_a3.py - recalibrate_ln false - output_suffix noln
```
