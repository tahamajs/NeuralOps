<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Ratio | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 1.00× | 183 MB | 1.9 s |
| DistilBERT | **ASVD** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_RATIO}}×** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 1.00× | 712 MB | 4.1 s |
| RoBERTa | **ASVD** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_RATIO}}×** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 1.00× | 793 MB | 4.2 s |
| GPT-2 | **ASVD** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_RATIO}}×** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After ASVD | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** | {{G_NOTE}} |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | ASVD (r=64) | Ratio |
|---|---|---|---|
| `q_lin` (768×768) | 589,824 | 98,368 | 6.0× |
| `k_lin` (768×768) | 589,824 | 98,368 | 6.0× |
| `v_lin` (768×768) | 589,824 | 98,368 | 6.0× |
| `out_lin` (768×768) | 589,824 | 98,368 | 6.0× |

## 7.3 Reconstruction Error Distribution

**Table 7.3: Frobenius reconstruction errors.**

| Layer type | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q | {{D_REC_Q}} | {{R_REC_Q}} | {{G_REC_Q}} |
| K | {{D_REC_K}} | {{R_REC_K}} | {{G_REC_K}} |
| V | {{D_REC_V}} | {{R_REC_V}} | {{G_REC_V}} |
| O | {{D_REC_O}} | {{R_REC_O}} | {{G_REC_O}} |
| **Mean** | **{{D_REC_MEAN}}** | **{{R_REC_MEAN}}** | **{{G_REC_MEAN}}** |

## 7.4 Rank Distribution

**Table 7.4: Energy-based selected rank per layer (DistilBERT).**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q_R}} | {{D_L0_K_R}} | {{D_L0_V_R}} | {{D_L0_O_R}} |
| 1 | {{D_L1_Q_R}} | {{D_L1_K_R}} | {{D_L1_V_R}} | {{D_L1_O_R}} |
| 2 | {{D_L2_Q_R}} | {{D_L2_K_R}} | {{D_L2_V_R}} | {{D_L2_O_R}} |
| 3 | {{D_L3_Q_R}} | {{D_L3_K_R}} | {{D_L3_V_R}} | {{D_L3_O_R}} |
| 4 | {{D_L4_Q_R}} | {{D_L4_K_R}} | {{D_L4_V_R}} | {{D_L4_O_R}} |
| 5 | {{D_L5_Q_R}} | {{D_L5_K_R}} | {{D_L5_V_R}} | {{D_L5_O_R}} |

**Observation.** {{RANK_OBS}}

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | ASVD | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | ASVD | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | ASVD | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.6 Ablation: Rank

**Table 7.6: Accuracy vs rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |
| 128 | {{D_R128}} | {{D_RATIO128}}× | {{R_R128}} | {{R_RATIO128}}× | {{G_R128}} | {{G_RATIO128}}× |

## 7.7 Ablation: Energy Threshold

**Table 7.7: Accuracy vs energy threshold.**

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_R}}× | {{R_E90}} | {{G_E90}} |
| 0.95 | {{D_E95}} | {{D_E95_R}}× | {{R_E95}} | {{G_E95}} |
| 0.99 | {{D_E99}} | {{D_E99_R}}× | {{R_E99}} | {{G_E99}} |
| 0.995 | {{D_E995}} | {{D_E995_R}}× | {{R_E995}} | {{G_E995}} |

## 7.8 Ablation: Dynamic vs Fixed Rank

**Table 7.8: Dynamic vs fixed rank.**

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dynamic ($\varepsilon$={{ENERGY_THRESHOLD}}) | {{D_DYN}} | {{R_DYN}} | {{G_DYN}} |
| Fixed (r={{RANK}}) | {{D_FIXED}} | {{R_FIXED}} | {{G_FIXED}} |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Accuracy vs LoRA rank.**

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 (no LoRA) | {{D_LORA0}} | {{R_LORA0}} | {{G_LORA0}} |
| 8 | {{D_LORA8}} | {{R_LORA8}} | {{G_LORA8}} |
| 16 | {{D_LORA16}} | {{R_LORA16}} | {{G_LORA16}} |
| 32 | {{D_LORA32}} | {{R_LORA32}} | {{G_LORA32}} |

## 7.10 Ablation: Error Threshold

**Table 7.10: Accuracy vs error threshold.**

| $\tau$ | DistilBERT | RoBERTa | GPT-2 | Success rate |
|---|---|---|---|---|
| 0.05 | {{D_T05}} | {{R_T05}} | {{G_T05}} | {{S05}}% |
| 0.10 | {{D_T10}} | {{R_T10}} | {{G_T10}} | {{S10}}% |
| 0.20 | {{D_T20}} | {{R_T20}} | {{G_T20}} | {{S20}}% |
| 0.30 | {{D_T30}} | {{R_T30}} | {{G_T30}} | {{S30}}% |

## 7.11 Comparison with Companion Methods

**Table 7.11: ASVD vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× | 79.36% | 1.00× |
| a3 (r=64) | {{A3_D}} | {{A3_D_R}}× | {{A3_R}} | {{A3_R_R}}× | {{A3_G}} | {{A3_G_R}}× |
| aasvd (r=288) | {{AASVD_D}} | {{AASVD_D_R}}× | {{AASVD_R}} | {{AASVD_R_R}}× | {{AASVD_G}} | {{AASVD_G_R}}× |
| **ASVD (r=64)** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** | **{{G_ACC}}** | **{{G_RATIO}}×** |

## 7.12 Failure Cases

**Table 7.12: Layers that aborted injection.**

| Layer | Model | Error | Reason |
|---|---|---|---|
| {{FAIL_LAYER_1}} | {{FAIL_MODEL_1}} | {{FAIL_ERR_1}} | {{FAIL_REASON_1}} |
| {{FAIL_LAYER_2}} | {{FAIL_MODEL_2}} | {{FAIL_ERR_2}} | {{FAIL_REASON_2}} |

## 7.13 Summary

| Metric | Value |
|---|---|
| Compression ratio | {{D_RATIO}}×–{{G_RATIO}}× |
| Accuracy drop | {{D_DELTA}}%–{{G_DELTA}}% |
| VRAM reduction | {{VRAM_REDUCTION}}% |
| Failures | {{N_FAILURES}} |
| Avg compression time | {{COMPRESS_TIME}} s |
```

---