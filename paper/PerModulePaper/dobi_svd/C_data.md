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
| Params after | {{D_PARAMS_FULL}} |
| Effective ratio | {{D_RATIO}}× |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| Nominal rank | {{RANK}} |
| Effective rank (τ_g=0.5) | {{D_REFF}} |
| Compressed layers | {{D_COMP}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_FULL}} |
| Effective ratio | {{R_RATIO}}× |
| Accuracy | {{R_ACC}}% |
| F1 | {{R_F1}} |
| Peak VRAM | {{R_VRAM}} MB |
| Nominal rank | {{RANK}} |
| Effective rank | {{R_EFF}} |
| Compressed layers | {{R_COMP}} |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_FULL}} |
| Effective ratio | {{G_RATIO}}× |
| Accuracy | {{G_ACC}}% |
| F1 | {{G_F1}} |
| Peak VRAM | {{G_VRAM}} MB |
| Nominal rank | {{RANK}} |
| Effective rank | {{G_EFF}} |
| Compressed layers | {{G_COMP}} |

## C.4 Per-Layer Gate Statistics (DistilBERT)

| Layer | Projection | Mean σ(α) | Std | % < 0.3 | % > 0.9 |
|---|---|---|---|---|---|
| 0 | q_lin | {{D_L0_Q_MEAN}} | {{D_L0_Q_STD}} | {{D_L0_Q_LOW}} | {{D_L0_Q_HIGH}} |
| 0 | k_lin | {{D_L0_K_MEAN}} | {{D_L0_K_STD}} | {{D_L0_K_LOW}} | {{D_L0_K_HIGH}} |
| 0 | v_lin | {{D_L0_V_MEAN}} | {{D_L0_V_STD}} | {{D_L0_V_LOW}} | {{D_L0_V_HIGH}} |
| 0 | out_lin | {{D_L0_O_MEAN}} | {{D_L0_O_STD}} | {{D_L0_O_LOW}} | {{D_L0_O_HIGH}} |
| ... | ... | ... | ... | ... | ... |
| 5 | out_lin | {{D_L5_O_MEAN}} | {{D_L5_O_STD}} | {{D_L5_O_LOW}} | {{D_L5_O_HIGH}} |

## C.5 Per-Layer Reconstruction Error

| Layer | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | {{D_L0_ERR}} | {{R_L0_ERR}} | {{G_L0_ERR}} |
| 1 | {{D_L1_ERR}} | {{R_L1_ERR}} | {{G_L1_ERR}} |
| ... | ... | ... | ... |
| 5/11 | {{D_LL_ERR}} | {{R_LL_ERR}} | {{G_LL_ERR}} |

## C.6 Rank Ablation

| $r$ | DistilBERT | Ratio | Effective $r_\text{eff}$ | RoBERTa | GPT-2 |
|---|---|---|---|---|---|
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{D_R64_REFF}} | {{R_R64}} | {{G_R64}} |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{D_R128_REFF}} | {{R_R128}} | {{G_R128}} |
| 192 | {{D_R192}} | {{D_RATIO192}}× | {{D_R192_REFF}} | {{R_R192}} | {{G_R192}} |
| 256 | {{D_R256}} | {{D_RATIO256}}× | {{D_R256_REFF}} | {{R_R256}} | {{G_R256}} |

## C.7 Init Gate Value Ablation

| α_init | σ(α_init) | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 1.0 | 0.731 | {{D_G1}} | {{R_G1}} | {{G_G1}} |
| 2.5 | 0.924 | {{D_G25}} | {{R_G25}} | {{G_G25}} |
| 4.0 | 0.982 | {{D_G4}} | {{R_G4}} | {{G_G4}} |
| 5.0 | 0.993 | {{D_G5}} | {{R_G5}} | {{G_G5}} |

## C.8 Pruning Threshold Ablation

| $\tau_g$ | $r_\text{eff}$ | Accuracy | Δ from unpruned |
|---|---|---|---|
| 0.1 | {{D_TAU1_REFF}} | {{D_TAU1_ACC}} | {{D_TAU1_D}} |
| 0.3 | {{D_TAU3_REFF}} | {{D_TAU3_ACC}} | {{D_TAU3_D}} |
| 0.5 | {{D_TAU5_REFF}} | {{D_TAU5_ACC}} | {{D_TAU5_D}} |
| 0.7 | {{D_TAU7_REFF}} | {{D_TAU7_ACC}} | {{D_TAU7_D}} |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss  Mean gate
1     {{D_S1}}  {{D_S1_GATE}}
5     {{D_S5}}  {{D_S5_GATE}}
10    {{D_S10}} {{D_S10_GATE}}
...
50    {{D_S50}} {{D_S50_GATE}}
```

### RoBERTa
```
Step  Loss  Mean gate
1     {{R_S1}}  {{R_S1_GATE}}
...
50    {{R_S50}} {{R_S50_GATE}}
```

### GPT-2
```
Step  Loss  Mean gate
1     {{G_S1}}  {{G_S1_GATE}}
...
50    {{G_S50}} {{G_S50_GATE}}
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": {{D_PARAMS_FULL}},
    "effective_ratio": {{D_RATIO}},
    "accuracy_pct": {{D_ACC}},
    "f1": {{D_F1}},
    "nominal_rank": {{RANK}},
    "effective_rank": {{D_REFF}},
    "mean_gate": {{D_GATE_MEAN}}
  },
  "roberta": {...},
  "gpt2": {...}
}
```

---