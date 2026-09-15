# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **SOA Fused** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_ATTN_PARAMS}} | **{{D_RATIO}}×** |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **SOA Fused** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_ATTN_PARAMS}} | **{{R_RATIO}}×** |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **SOA Fused** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_ATTN_PARAMS}} | **{{G_RATIO}}×** |

**Key observations:**

1. **Attention parameters reduce by {{D_RATIO}}×** (DistilBERT at $k = {{RANK}}$).
2. **Accuracy drop is only {{D_DELTA}}%** — recoverable.
3. **VRAM decreases by {{VRAM_REDUCTION}}%.**

## 7.2 Parameter Analysis

**Table 7.2: Total parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_TOTAL_PARAMS}} | {{D_TOTAL_DELTA}}% |
| RoBERTa | 124,647,170 | {{R_TOTAL_PARAMS}} | {{R_TOTAL_DELTA}}% |
| GPT-2 | 124,441,344 | {{G_TOTAL_PARAMS}} | {{G_TOTAL_DELTA}}% |

**Per-projection breakdown (DistilBERT):**

| Projection | Baseline | Ours | Ratio |
|---|---|---|---|
| $W_Q$ | 589,824 | {{D_Q}} | {{D_Q_RATIO}}× |
| $W_K$ | 589,824 | {{D_K}} | {{D_K_RATIO}}× |
| $W_V$ | 589,824 | {{D_V}} | {{D_V_RATIO}}× |
| $W_O$ | 589,824 | {{D_O}} | {{D_O_RATIO}}× |
| **Total** | 2,359,296 | {{D_ATTN_PARAMS}} | **{{D_RATIO}}×** |

## 7.3 Reconstruction Error

**Table 7.3: Frobenius reconstruction errors per projection.**

| Model | Q | K | V |
|---|---|---|---|
| DistilBERT | {{D_REC_Q}} | {{D_REC_K}} | {{D_REC_V}} |
| RoBERTa | {{R_REC_Q}} | {{R_REC_K}} | {{R_REC_V}} |
| GPT-2 | {{G_REC_Q}} | {{G_REC_K}} | {{G_REC_V}} |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Baseline Time | Ours | VRAM |
|---|---|---|---|
| DistilBERT | 1.90 s | {{D_TIME}} s | {{D_VRAM}} MB |
| RoBERTa | 4.10 s | {{R_TIME}} s | {{R_VRAM}} MB |
| GPT-2 | 4.20 s | {{G_TIME}} s | {{G_VRAM}} MB |

## 7.5 Ablation: Rank $k$

**Table 7.5: Accuracy vs bottleneck rank.**

| $k$ | Ratio | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 12× | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | 6× | {{D_R64}} | {{R_R64}} | {{G_R64}} |
| 128 | 3× | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| 192 | 2× | {{D_R192}} | {{R_R192}} | {{G_R192}} |
| 384 | 1× | {{D_R384}} | {{R_R384}} | {{G_R384}} |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| LoRA $r_L$ | DistilBERT | RoBERTa | GPT-2 |
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

## 7.8 Training Loss

**Figure 7.1** shows recovery loss vs step. {{TRAINING_DESC}}

## 7.9 Summary

- **Compression:** {{D_RATIO}}×–{{G_RATIO}}× on attention projections.
- **Accuracy:** {{D_DELTA}}%–{{G_DELTA}}% drop.
- **VRAM:** {{VRAM_REDUCTION}}% reduction.
- **No calibration data required.**

---
