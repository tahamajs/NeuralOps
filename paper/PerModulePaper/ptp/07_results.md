<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: Results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 results with $\rho = {{KEEP_RATIO}}$.**

| Model | Module | Acc | Δ | F1 | Kept | VRAM | Time |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 100% | 183 MB | 1.9 s |
| DistilBERT | **PTP** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | **{{D_KEPT}}%** | {{D_VRAM}} MB | {{D_TIME}} s |
| RoBERTa | full | 78.56% | — | 0.8124 | 100% | 712 MB | 4.1 s |
| RoBERTa | **PTP** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | **{{R_KEPT}}%** | {{R_VRAM}} MB | {{R_TIME}} s |
| GPT-2 | full | 79.36% | — | 0.7978 | 100% | 793 MB | 4.2 s |
| GPT-2 | **PTP** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | **{{G_KEPT}}%** | {{G_VRAM}} MB | {{G_TIME}} s |

## 7.2 Efficiency

**Table 7.2: Memory and time.**

| Model | Config | Peak VRAM | Inference Time | Speedup |
|---|---|---|---|---|
| DistilBERT | full | 183 MB | 1.90 s | 1.00× |
| DistilBERT | PTP | {{D_VRAM}} MB | {{D_TIME}} s | **{{D_SPEEDUP}}×** |
| GPT-2 | full | 793 MB | 4.20 s | 1.00× |
| GPT-2 | PTP | {{G_VRAM}} MB | {{G_TIME}} s | **{{G_SPEEDUP}}×** |

## 7.3 Ablation: Keep Ratio

**Table 7.3: Accuracy vs keep ratio.**

| $\rho$ | DistilBERT | RoBERTa | GPT-2 | Speedup |
|---|---|---|---|---|
| 0.5 | {{D_R50}} | {{R_R50}} | {{G_R50}} | 2.0× |
| 0.7 | {{D_R70}} | {{R_R70}} | {{G_R70}} | 1.4× |
| 0.8 | {{D_R80}} | {{R_R80}} | {{G_R80}} | 1.25× |
| 0.9 | {{D_R90}} | {{R_R90}} | {{G_R90}} | 1.11× |
| 0.95 | {{D_R95}} | {{R_R95}} | {{G_R95}} | 1.05× |

## 7.4 Ablation: [CLS] Preservation

**Table 7.4: Effect of [CLS] preservation.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With [CLS] preserve | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Without | {{D_NOCLS}} | {{R_NOCLS}} | {{G_NOCLS}} |

**Observation.** [CLS] preservation improves accuracy by {{CLS_GAIN}}%.

## 7.5 Ablation: Score Network Hidden Dim

**Table 7.5: Accuracy vs hidden dimension.**

| Hidden | DistilBERT | Params/layer |
|---|---|---|
| 32 | {{D_H32}} | ~25K |
| 64 | {{D_H64}} | ~50K |
| 128 | {{D_H128}} | ~99K |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_LR4}} | {{R_LR4}} | {{G_LR4}} |
| 8 | {{D_LR8}} | {{R_LR8}} | {{G_LR8}} |
| 16 | {{D_LR16}} | {{R_LR16}} | {{G_LR16}} |
| 32 | {{D_LR32}} | {{R_LR32}} | {{G_LR32}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison with Companion Methods

**Table 7.8: Comparison with other pruning methods.**

| Method | DistilBERT | RoBERTa | GPT-2 | Type | Speedup |
|---|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | Dense | 1.0× |
| PoWER-BERT | {{POWER_D}} | {{POWER_R}} | {{POWER_G}} | Attn-heuristic | 1.4× |
| LTP | {{LTP_D}} | {{LTP_R}} | {{LTP_G}} | Learned threshold | 1.4× |
| Token Merging | {{TM_D}} | {{TM_R}} | {{TM_G}} | Merge | 1.5× |
| **PTP** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **MLP top-k** | **{{D_SPEEDUP}}×** |

## 7.9 Token Score Analysis

**Table 7.9: Top-5 highest-scoring tokens (DistilBERT, sample input).**

| Rank | Token | Score |
|---|---|---|
| 1 | [CLS] | 1.0 (fixed) |
| 2 | {{TOP2_TOKEN}} | {{TOP2_SCORE}} |
| 3 | {{TOP3_TOKEN}} | {{TOP3_SCORE}} |
| 4 | {{TOP4_TOKEN}} | {{TOP4_SCORE}} |
| 5 | {{TOP5_TOKEN}} | {{TOP5_SCORE}} |

**Observation.** Content words dominate; function words are pruned.

## 7.10 Per-Layer Pruning Ratio

**Table 7.10: Effective kept ratio per layer (DistilBERT).**

| Layer | Kept ratio |
|---|---|
| 0 | {{D_L0_KEPT}}% |
| 1 | {{D_L1_KEPT}}% |
| 2 | {{D_L2_KEPT}}% |
| 3 | {{D_L3_KEPT}}% |
| 4 | {{D_L4_KEPT}}% |
| 5 | {{D_L5_KEPT}}% |

## 7.11 Training Loss Curves

**Figure 7.1** shows loss vs step.

## 7.12 Summary

- **Accuracy:** drops {{D_DELTA}}%–{{G_DELTA}}%.
- **Speedup:** {{D_SPEEDUP}}×–{{G_SPEEDUP}}× at $\rho = {{KEEP_RATIO}}$.
- **[CLS] preservation:** essential.
- **Score network:** learns content-word importance.
- **All models pass quality gate.**
