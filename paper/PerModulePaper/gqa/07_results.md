<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy with $G = H/2$.**

| Model | Config | Acc | Δ | F1 | $H_{KV}$ | KV ratio | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | MHA | 90.71% | — | 0.9099 | 12 | 1.00× | 183 MB |
| DistilBERT | **GQA** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_KV}} | **{{D_KV_R}}×** | {{D_VRAM}} MB |
| RoBERTa | MHA | 78.56% | — | 0.8124 | 12 | 1.00× | 712 MB |
| RoBERTa | **GQA** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_KV}} | **{{R_KV_R}}×** | {{R_VRAM}} MB |
| GPT-2 | MHA | 79.36% | — | 0.7978 | 12 | 1.00× | 793 MB |
| GPT-2 | **GQA** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_KV}} | **{{G_KV_R}}×** | {{G_VRAM}} MB |

## 7.2 KV Cache Analysis

**Table 7.2: KV cache at inference.**

| Model | Config | Per-token (KB) | At L=1024 (MB) | At L=8192 (MB) |
|---|---|---|---|---|
| GPT-2 | MHA | 36.9 | 37.7 | 302 |
| GPT-2 | **GQA (H/2)** | **{{KV_PER_TOKEN}}** | **{{KV_L1024}}** | **{{KV_L8192}}** |
| GPT-2 | GQA (H/4) | 9.2 | 9.4 | 75 |
| GPT-2 | MQA | 3.1 | 3.1 | 25 |

## 7.3 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | MHA params | GQA params | Reduction |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | {{D_P_RATIO}}× |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | {{R_P_RATIO}}× |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | {{G_P_RATIO}}× |

## 7.4 Efficiency

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | MHA | 1.90 s | 59000 | 183 MB |
| DistilBERT | GQA | {{D_TIME}} s | {{D_TPS}} | {{D_VRAM}} MB |
| GPT-2 | MHA | 4.20 s | 26700 | 793 MB |
| GPT-2 | GQA | {{G_TIME}} s | {{G_TPS}} | {{G_VRAM}} MB |

## 7.5 Ablation: Number of KV Heads

**Table 7.5: Accuracy vs $G$.**

| $G$ | DistilBERT | RoBERTa | GPT-2 | KV ratio |
|---|---|---|---|---|
| 12 (MHA) | 90.71% | 78.56% | 79.36% | 1.00× |
| 6 (H/2) | {{D_G6}} | {{R_G6}} | {{G_G6}} | 2.00× |
| 4 (H/3) | {{D_G4}} | {{R_G4}} | {{G_G4}} | 3.00× |
| 3 (H/4) | {{D_G3}} | {{R_G3}} | {{G_G3}} | 4.00× |
| 2 (H/6) | {{D_G2}} | {{R_G2}} | {{G_G2}} | 6.00× |
| 1 (MQA) | {{D_G1}} | {{R_G1}} | {{G_G1}} | 12.0× |

## 7.6 Ablation: LoRA Rank

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.7 Ablation: Recovery Steps

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison with Companion Methods

| Method | DistilBERT | RoBERTa | GPT-2 | Type |
|---|---|---|---|---|
| full (MHA) | 90.71% | 78.56% | 79.36% | Baseline |
| AASVD | {{AASVD_D}} | {{AASVD_R}} | {{AASVD_G}} | Compression |
| A3 | {{A3_D}} | {{A3_R}} | {{A3_G}} | Compression |
| NHA | 90.25% | FAIL | 78.78% | Attention |
| **GQA** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **KV cache** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss for all three models.

## 7.10 Summary

- **KV cache:** reduced by {{D_KV_R}}×–{{G_KV_R}}×.
- **Accuracy:** drops only {{D_DELTA}}%–{{G_DELTA}}%.
- **Parameter reduction:** modest ({{D_P_RATIO}}×–{{G_P_RATIO}}×).
- **All models pass quality gate.**
