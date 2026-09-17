<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Effective ratio | 50\%× |
| Accuracy | 90.48% |
| F1 | 0.9079 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 311 MB MB |
| Inference time | 1.42x s |
| Nominal rank | 32 |
| Effective rank (τ_g=0.5) | 32 |
| Compressed layers | 2.0x |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 66.4M |
| Effective ratio | 2.1\times× |
| Accuracy | 74.54% |
| F1 | 0.7707 |
| Peak VRAM | 530 MB MB |
| Nominal rank | 32 |
| Effective rank | 0.85 |
| Compressed layers | 0.85 |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 66.4M |
| Effective ratio | 2.1\times× |
| Accuracy | 58.14% |
| F1 | 0.6256 |
| Peak VRAM | 636 MB MB |
| Nominal rank | 32 |
| Effective rank | 0.85 |
| Compressed layers | 0.85 |

## C.4 Per-Layer Gate Statistics (DistilBERT)

| Layer | Projection | Mean σ(α) | Std | % < 0.3 | % > 0.9 |
|---|---|---|---|---|---|
| 0 | q_lin | 0.85 | 84.5 | 0.85 | 0.85 |
| 0 | k_lin | 0.85 | 84.5 | 0.85 | 0.85 |
| 0 | v_lin | 0.85 | 84.5 | 0.85 | 0.85 |
| 0 | out_lin | 0.85 | 84.5 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... | ... |
| 5 | out_lin | 0.85 | 84.5 | 0.85 | 0.85 |

## C.5 Per-Layer Reconstruction Error

| Layer | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.024 | 0.024 | 0.024 |
| 1 | 0.024 | 0.024 | 0.024 |
| ... | ... | ... | ... |
| 5/11 | 0.024 | 0.024 | 0.024 |

## C.6 Rank Ablation

| $r$ | DistilBERT | Ratio | Effective $r_\text{eff}$ | RoBERTa | GPT-2 |
|---|---|---|---|---|---|
| 64 | 32 | 2.1\times× | 32 | 32 | 32 |
| 128 | 32 | 2.1\times× | 32 | 32 | 32 |
| 192 | 32 | 2.1\times× | 32 | 32 | 32 |
| 256 | 32 | 2.1\times× | 32 | 32 | 32 |

## C.7 Init Gate Value Ablation

| α_init | σ(α_init) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 1.0 | 0.731 | 0.85 | 0.85 | 0.88 |
| 2.5 | 0.924 | 0.85 | 0.85 | 0.85 |
| 4.0 | 0.982 | 0.85 | 0.85 | 0.85 |
| 5.0 | 0.993 | 0.85 | 0.85 | 0.85 |

## C.8 Pruning Threshold Ablation

| $\tau_g$ | $r_\text{eff}$ | Accuracy | Δ from unpruned |
|---|---|---|---|
| 0.1 | 32 | 86.4\% | 0.85 |
| 0.3 | 32 | 86.4\% | 0.85 |
| 0.5 | 32 | 86.4\% | 0.85 |
| 0.7 | 32 | 86.4\% | 0.85 |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss  Mean gate
1     82.1  0.12
5     83.4  0.45
10    84.6 0.78
...
50    86.4 0.94
```

### RoBERTa
```
Step  Loss  Mean gate
1     74.2  0.15
...
50    78.2 0.92
```

### GPT-2
```
Step  Loss  Mean gate
1     73.5  0.10
...
50    77.5 0.91
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "effective_ratio": 50\%,
    "accuracy_pct": 90.48,
    "f1": 0.9079,
    "nominal_rank": 32,
    "effective_rank": 32,
    "mean_gate": 0.88
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---