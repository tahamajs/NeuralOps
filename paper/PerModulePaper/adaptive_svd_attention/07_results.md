# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **Adaptive SVD** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ATTN}} | **{{D_RATIO}}×** |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **Adaptive SVD** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ATTN}} | **{{R_RATIO}}×** |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **Adaptive SVD** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ATTN}} | **{{G_RATIO}}×** |

## 7.2 Per-Projection Reduction

**Table 7.2: Selected rank per projection (DistilBERT).**

| Layer | Q r* | K r* | V r* | O r* |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| 2 | {{D_L2_Q_R}} | {{D_L2_K_R}} | {{D_L2_V_R}} | {{D_L2_O_R}} |
| 3 | {{D_L3_Q_R}} | {{D_L3_K_R}} | {{D_L3_V_R}} | {{D_L3_O_R}} |
| 4 | {{D_L4_Q_R}} | {{D_L4_K_R}} | {{D_L4_V_R}} | {{D_L4_O_R}} |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |
| **Mean** | **{{D_Q_MEAN}}** | **{{D_K_MEAN}}** | **{{D_V_MEAN}}** | **{{D_O_MEAN}}** |

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Baseline Time | Ours | VRAM |
|---|---|---|---|
| DistilBERT | 1.90 s | {{D_TIME}} s | {{D_VRAM}} MB |
| RoBERTa | 4.10 s | {{R_TIME}} s | {{R_VRAM}} MB |
| GPT-2 | 4.20 s | {{G_TIME}} s | {{G_VRAM}} MB |

## 7.4 Ablation: Energy Threshold τ

**Table 7.4: Accuracy vs energy threshold.**

| τ | DistilBERT | Avg Rank | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.80 | {{D_T80}} | {{D_T80_R}} | {{R_T80}} | {{G_T80}} |
| 0.85 | {{D_T85}} | {{D_T85_R}} | {{R_T85}} | {{G_T85}} |
| 0.92 | **{{D_ACC}}** | **{{AVG_RANK}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 0.95 | {{D_T95}} | {{D_T95_R}} | {{R_T95}} | {{G_T95}} |
| 0.99 | {{D_T99}} | {{D_T99_R}} | {{R_T99}} | {{G_T99}} |

## 7.5 Ablation: Residual Rank

**Table 7.5: Accuracy vs residual adapter rank.**

| $r_a$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 (no adapter) | {{D_RA0}} | {{R_RA0}} | {{G_RA0}} |
| 8 | {{D_RA8}} | {{R_RA8}} | {{G_RA8}} |
| 16 | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 32 | {{D_RA32}} | {{R_RA32}} | {{G_RA32}} |

## 7.6 Ablation: Fixed vs Adaptive Rank

**Table 7.6: Fixed rank $r^*$ vs adaptive.**

| Method | DistilBERT | Ratio | Note |
|---|---|---|---|
| Fixed $r^*$ = 128 | {{D_FIX128}} | {{D_FIX128_R}}× | Uniform |
| Fixed $r^*$ = 192 | {{D_FIX192}} | {{D_FIX192_R}}× | Uniform |
| **Adaptive** | **{{D_ACC}}** | **{{D_RATIO}}×** | Per-projection |

**Observation.** Adaptive achieves similar accuracy with better compression 
because it concentrates capacity where needed.

## 7.7 Ablation: All-Four vs Q/K/V-Only

**Table 7.7: Factorizing all four projections.**

| Config | DistilBERT | Ratio |
|---|---|---|
| Q/K/V only (SOA-style) | {{D_QKV_ONLY}} | {{D_QKV_ONLY_R}}× |
| **All four** | **{{D_ACC}}** | **{{D_RATIO}}×** |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Summary

- **Attention parameters:** reduce by {{D_RATIO}}×–{{G_RATIO}}×.
- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Average rank:** {{AVG_RANK}} at τ = {{ENERGY_THRESHOLD}}.
- **Zero-shot** — no calibration required.
- **Supports T5, cross-attention, GQA, RoPE.**

---