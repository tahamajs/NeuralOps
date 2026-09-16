<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Path | Time | Status |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | Full | 1.9 s | PASS |
| DistilBERT | **FASA** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | Pruned | {{D_TIME}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | Full | 4.1 s | PASS |
| RoBERTa | **FASA** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | Pruned | {{R_TIME}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | Full | 4.2 s | PASS |
| GPT-2 | **FASA** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | **No prune** | {{G_TIME}} | {{G_STATUS}} |

**Key observations:**

1. DistilBERT: {{D_OBS}}
2. RoBERTa: {{R_OBS}}
3. GPT-2: {{G_OBS}} — decoder path is a **no-op** as designed.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | FASA | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**FASA adds only LoRA parameters** (no learned pruning model).

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (FASA) | Ratio | VRAM (base) | VRAM (FASA) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Keep Ratio

**Table 7.4: Accuracy vs keep ratio.**

| Keep ratio | DistilBERT | RoBERTa | Kept tokens (L=128) |
|---|---|---|---|
| 0.25 | {{D_K25}} | {{R_K25}} | 32 |
| 0.50 | {{D_K50}} | {{R_K50}} | 64 |
| 0.75 | {{D_K75}} | {{R_K75}} | 96 |
| 1.00 | {{D_K100}} | {{R_K100}} | 128 (baseline) |

**Observation.** {{KEEP_RATIO_OBS}}

## 7.5 Ablation: Importance Method

**Table 7.5: Variance vs L2 norm.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `var` | {{D_VAR}} | {{R_VAR}} | N/A (no prune) |
| `l2` | {{D_L2}} | {{R_L2}} | N/A (no prune) |

**Observation.** {{IMP_METHOD_OBS}}

## 7.6 Ablation: CLS Protection

**Table 7.6: Effect of `protect_first_token`.**

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| True | {{D_PROTECT_ON}} | {{R_PROTECT_ON}} |
| False | {{D_PROTECT_OFF}} | {{R_PROTECT_OFF}} |

**Observation.** {{PROTECT_OBS}}

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank (keep=0.5).**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.8 Ablation: Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | {{COS_D}} | {{COS_R}} | {{G_ACC}} | +5.7% |
| nha | 90.25% | OOM | 78.78% | +0.88% |
| a3 | {{A3_D}} | {{A3_R}} | {{A3_G}} | **-60%** |
| **fasa** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.11 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}}) — {{D_VERDICT}}
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}}) — {{R_VERDICT}}
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}}) — **no-op** (decoder path skips pruning)
```

---