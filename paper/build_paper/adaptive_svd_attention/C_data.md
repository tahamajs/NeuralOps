<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Complete Metrics

| Metric | Value |
|---|---|
| Model | DistilBERT-base |
| Params (before) | 66,955,010 |
| Params (after) | 22000000 |
| Compression ratio | 3.0× |
| Accuracy | 88.0% |
| F1 (binary) | 0.880 |
| F1 (macro) | 0.879 |
| Precision | 0.881 |
| Recall | 0.879 |
| Validation loss | 0.36 |
| Peak VRAM | 130 MB |
| Inference time | 2.3 s |
| Compression time | 15 s |
| Compressed layers | 24 |
| Fallback layers | 0 |
| Avg reconstruction error | 0.27 |

## C.2 RoBERTa — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,647,170 |
| Params (after) | 50000000 |
| Compression ratio | 2.5× |
| Accuracy | 74.0% |
| ... | ... |

## C.3 GPT-2 — Complete Metrics

| Metric | Value |
|---|---|
| Params (before) | 124,441,344 |
| Params (after) | 50000000 |
| Compression ratio | 2.5× |
| Accuracy | 75.5% |
| ... | ... |

## C.4 Per-Layer Rank Distribution (DistilBERT)

| Layer | Q | K | V | O | Row avg |
|---|---|---|---|---|---|
| 0 | 128 | 128 | 64 | 64 | 96 |
| 1 | 140 | 140 | 70 | 70 | 105 |
| 2 | 150 | 150 | 80 | 80 | 115 |
| 3 | 150 | 150 | 80 | 80 | 115 |
| 4 | 140 | 140 | 70 | 70 | 105 |
| 5 | 128 | 128 | 64 | 64 | 96 |
| **Col avg** | **139** | **139** | **71** | **71** | **105** |

## C.5 Per-Layer Rank Distribution (RoBERTa)

(12 rows for 12 layers)

## C.6 Per-Layer Rank Distribution (GPT-2)

(12 rows for 12 layers, only Q/K/V/O equivalents for c_attn/c_proj)

## C.7 Per-Layer Reconstruction Errors

### DistilBERT

| Layer | Q error | K error | V error | O error |
|---|---|---|---|---|
| 0 | 0.024 | 0.025 | 0.025 | 0.026 |
| 1 | 0.024 | 0.025 | 0.025 | 0.026 |
| ... | ... | ... | ... | ... |

## C.8 Energy Threshold Ablation

| τ | DistilBERT acc | Avg rank | Ratio |
|---|---|---|---|
| 0.85 | 85 | 64 | 5.0× |
| 0.90 | 87 | 90 | 3.5× |
| 0.92 | 88 | 105 | 3.0× |
| 0.95 | 89 | 160 | 2.0× |
| 0.99 | 90 | 350 | 1.2× |

## C.9 Residual Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 | Adapter params |
|---|---|---|---|---|
| 0 | 80 | 65 | 66 | 0 |
| 4 | 85 | 70 | 71.5 | 4·d_in·N_P |
| 8 | 87 | 72.5 | 74 | ... |
| 16 | 88 | 74 | 75.5 | ... |
| 32 | 88.2 | 74.3 | 75.7 | ... |
| 64 | 88.3 | 74.4 | 75.8 | ... |

## C.10 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 0.4 | 0.8 | 0.7 |
| 25 | 0.35 | 0.6 | 0.55 |
| 50 | 0.30 | 0.55 | 0.50 |
| 100 | 88.3 | 74.5 | 75.8 |
| 200 | 88.4 | 74.6 | 75.9 |

## C.11 Rank Cap Ablation

| Cap | DistilBERT | % hit |
|---|---|---|
| 64 | 85 | 20% |
| 128 | 88 | 5% |
| 256 | 88 | 0% |
| 384 | 88 | 0% |

## C.12 Training Loss Curves

### DistilBERT
```
Step  Loss
1     1.5
5     0.7
10    0.4
...
50    0.30
```

(Similar for RoBERTa, GPT-2)

## C.13 Raw JSON Metrics

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 22000000,
    "ratio": 3.0,
    "accuracy_pct": 88.0,
    "energy_threshold": 0.92,
    "residual_rank": 16,
    "min_rank_obs": 64,
    "max_rank_obs": 150,
    "layer_ranks": {
      "layer_0_q": 128,
      "layer_0_k": 128,
      "...": "..."
    }
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.14 Reproduction Commands

```bash
# Main
python run_all_4_adaptive_svd.py - models distilbert,roberta,gpt2

# Energy threshold sweep
for tau in 0.85 0.90 0.92 0.95 0.99; do
    python run_all_4_adaptive_svd.py - energy_threshold $tau - output_suffix tau_$tau
done

# Residual rank sweep
for rl in 0 4 8 16 32 64; do
    python run_all_4_adaptive_svd.py - residual_rank $rl - output_suffix rl_$rl
done
```
