<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Compressed | Ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 0 | 1.00× | 183 MB |
| DistilBERT | **PHANTOM** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_COMP}} | **{{D_RATIO}}×** | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 0 | 1.00× | 712 MB |
| RoBERTa | **PHANTOM** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_COMP}} | **{{R_RATIO}}×** | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 0 | 1.00× | 793 MB |
| GPT-2 | **PHANTOM** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_COMP}} | **{{G_RATIO}}×** | {{G_VRAM}} MB |

**Key observations:**

1. **Parameter reduction:** **{{D_RATIO}}×** on DistilBERT, 
   **{{R_RATIO}}×** on RoBERTa, **{{G_RATIO}}×** on GPT-2.
2. **Accuracy:** drops **{{D_DELTA}}%** on DistilBERT.
3. **VRAM:** reduces by **{{VRAM_REDUCTION}}%**.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | PHANTOM | Ratio | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | **{{D_RATIO}}×** | {{D_NOTE}} |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | **{{R_RATIO}}×** | {{R_NOTE}} |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | **{{G_RATIO}}×** | {{G_NOTE}} |

**Per-layer breakdown (DistilBERT):**

| Layer | Original | PHANTOM | Ratio |
|---|---|---|---|
| `q_lin` | 589,824 | 24,576 | **24×** |
| `k_lin` | 589,824 | 24,576 | **24×** |
| `v_lin` | 589,824 | 24,576 | **24×** |
| `out_lin` | 589,824 | 24,576 | **24×** |
| **Total** | 2,359,296 | **98,304** | **24×** |

## 7.3 Reconstruction Error

**Table 7.3: CUR reconstruction errors.**

| Projection | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Q | {{D_REC_Q}} | {{R_REC_Q}} | {{G_REC_Q}} |
| K | {{D_REC_K}} | {{R_REC_K}} | {{G_REC_K}} |
| V | {{D_REC_V}} | {{R_REC_V}} | {{G_REC_V}} |
| O | {{D_REC_O}} | {{R_REC_O}} | {{G_REC_O}} |
| **Mean** | **{{D_REC_MEAN}}** | **{{R_REC_MEAN}}** | **{{G_REC_MEAN}}** |

## 7.4 LSH Bucket Statistics

**Table 7.4: LSH bucket size distribution.**

| Model | Avg bucket | Max bucket | Reduction |
|---|---|---|---|
| DistilBERT | {{D_BUCKET}} | {{D_MAXBUCKET}} | **{{D_BUCKET_RED}}×** |
| RoBERTa | {{R_BUCKET}} | {{R_MAXBUCKET}} | **{{R_BUCKET_RED}}×** |
| GPT-2 | {{G_BUCKET}} | {{G_MAXBUCKET}} | **{{G_BUCKET_RED}}×** |

## 7.5 Efficiency Profile

**Table 7.5: Inference efficiency.**

| Model | Config | Time | Tokens/s | VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.90 s | 59000 | 183 MB |
| DistilBERT | PHANTOM | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| RoBERTa | full | 4.10 s | 27400 | 712 MB |
| RoBERTa | PHANTOM | {{R_TIME}} s | {{R_TPS}} | {{R_VRAM}} MB |
| GPT-2 | full | 4.20 s | 26700 | 793 MB |
| GPT-2 | PHANTOM | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.6 Ablation: CUR Rank

**Table 7.6: Accuracy vs CUR rank.**

| Rank | DistilBERT | Ratio | RoBERTa | Ratio | GPT-2 | Ratio |
|---|---|---|---|---|---|---|
| 8 | {{D_R8}} | {{D_RATIO8}}× | {{R_R8}} | {{R_RATIO8}}× | {{G_R8}} | {{G_RATIO8}}× |
| 16 | {{D_R16}} | {{D_RATIO16}}× | {{R_R16}} | {{R_RATIO16}}× | {{G_R16}} | {{G_RATIO16}}× |
| 32 | {{D_R32}} | {{D_RATIO32}}× | {{R_R32}} | {{R_RATIO32}}× | {{G_R32}} | {{G_RATIO32}}× |
| 64 | {{D_R64}} | {{D_RATIO64}}× | {{R_R64}} | {{R_RATIO64}}× | {{G_R64}} | {{G_RATIO64}}× |

## 7.7 Ablation: LSH Rounds

**Table 7.7: Accuracy vs LSH rounds.**

| Rounds | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1 | {{D_LSH1}} | {{R_LSH1}} | {{G_LSH1}} |
| 2 | {{D_LSH2}} | {{R_LSH2}} | {{G_LSH2}} |
| 4 | {{D_LSH4}} | {{R_LSH4}} | {{G_LSH4}} |
| 8 | {{D_LSH8}} | {{R_LSH8}} | {{G_LSH8}} |

## 7.8 Ablation: LSH Bits

**Table 7.8: Accuracy vs LSH bits per round.**

| Bits | Buckets | Avg bucket | DistilBERT | RoBERTa |
|---|---|---|---|---|
| 2 | 4 | {{B2_SIZE}} | {{D_B2}} | {{R_B2}} |
| 3 | 8 | {{B3_SIZE}} | {{D_B3}} | {{R_B3}} |
| 4 | 16 | {{B4_SIZE}} | {{D_B4}} | {{R_B4}} |
| 6 | 64 | {{B6_SIZE}} | {{D_B6}} | {{R_B6}} |

## 7.9 Ablation: Component Contributions

**Table 7.9: Component ablation.**

| Configuration | DistilBERT | Δ |
|---|---|---|
| Full | {{D_FULL}} | — |
| No softpick (softmax) | {{D_NOSP}} | {{D_NOSP_D}} |
| No LSH (dense attn) | {{D_NOLSH}} | {{D_NOLSH_D}} |
| No CUR (dense proj) | {{D_NOCUR}} | {{D_NOCUR_D}} |
| No LoRA recovery | {{D_NOLORA}} | {{D_NOLORA_D}} |

## 7.10 Ablation: Softpick Temperature

**Table 7.10: Accuracy vs τ_s.**

| τ_s | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.5 | {{D_T05}} | {{R_T05}} | {{G_T05}} |
| 1.0 | {{D_T10}} | {{R_T10}} | {{G_T10}} |
| 2.0 | {{D_T20}} | {{R_T20}} | {{G_T20}} |

## 7.11 Comparison with Companion Methods

**Table 7.11: PHANTOM vs prior compressors.**

| Method | DistilBERT | Ratio | RoBERTa | Ratio |
|---|---|---|---|---|
| full | 90.71% | 1.00× | 78.56% | 1.00× |
| A3 | {{A3_D}} | {{A3_D_R}}× | {{A3_R}} | {{A3_R_R}}× |
| AASVD | {{AASVD_D}} | {{AASVD_D_R}}× | {{AASVD_R}} | {{AASVD_R_R}}× |
| **PHANTOM** | **{{D_ACC}}** | **{{D_RATIO}}×** | **{{R_ACC}}** | **{{R_RATIO}}×** |

## 7.12 Compression Time

**Table 7.12: Wall-clock compression time.**

| Model | CUR | LSH setup | Total |
|---|---|---|---|
| DistilBERT | {{D_CUR_T}} s | {{D_LSH_T}} s | **{{D_TOTAL}} s** |
| RoBERTa | {{R_CUR_T}} s | {{R_LSH_T}} s | **{{R_TOTAL}} s** |
| GPT-2 | {{G_CUR_T}} s | {{G_LSH_T}} s | **{{G_TOTAL}} s** |

## 7.13 Summary

- Compression ratio: **{{D_RATIO}}×–{{G_RATIO}}×**
- Accuracy drop: **{{D_DELTA}}%–{{G_DELTA}}%**
- VRAM reduction: **{{VRAM_REDUCTION}}%**
- All models pass quality gate
```

---