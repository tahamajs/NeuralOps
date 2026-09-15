# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 |
| DistilBERT | **HGRN2-GDN** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 |
| RoBERTa | **HGRN2-GDN** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 |
| GPT-2 | **HGRN2-GDN** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**We state explicitly:** HGRN2-GatedDeltaNet **increases** parameter count.

## 7.3 Inference Speed vs Sequence Length

**Table 7.3: Inference time (seconds) vs sequence length (DistilBERT).**

| L | Baseline | Ours | Speedup |
|---|---|---|---|
| 128 | 1.90 | {{D_T128}} | {{S128}}× |
| 256 | 3.50 | {{D_T256}} | {{S256}}× |
| 512 | 12.0 | {{D_T512}} | {{S512}}× |
| 1024 | 45.0 | {{D_T1024}} | {{S1024}}× |
| 2048 | 180.0 | {{D_T2048}} | {{S2048}}× |

**Breakeven point:** $L^* = {{BREAKEVEN}}$.

**Important:** At L = 128, our method is **{{SLOWDOWN}}× slower** than 
baseline due to the Python-level sequential recurrence. At L ≥ 512, our 
method is faster.

## 7.4 Training Time

**Table 7.4: Wall-clock training time.**

| Model | Baseline (s/step) | Ours (s/step) | Slowdown |
|---|---|---|---|
| DistilBERT | 0.05 | {{D_TRAIN}} | {{D_TRAIN_R}}× |
| RoBERTa | 0.08 | {{R_TRAIN}} | {{R_TRAIN_R}}× |
| GPT-2 | 0.08 | {{G_TRAIN}} | {{G_TRAIN_R}}× |

## 7.5 Ablation: Gate Initialization

**Table 7.5: Accuracy vs gate bias.**

| Gate bias | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| -4 (favor DeltaNet) | {{D_GB_M4}} | {{R_GB_M4}} | {{G_GB_M4}} |
| 0 (equal blend) | {{D_GB_0}} | {{R_GB_0}} | {{G_GB_0}} |
| +4 (favor HGRN) | {{D_GB_P4}} | {{R_GB_P4}} | {{G_GB_P4}} |

## 7.6 Ablation: Branch Choice

**Table 7.6: Effect of using a single branch.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| HGRN-2 only | {{D_HGRN}} | {{R_HGRN}} | {{G_HGRN}} |
| DeltaNet only | {{D_DN}} | {{R_DN}} | {{G_DN}} |
| Both (learned gate) | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** |
| Both (fixed gate 0.5) | {{D_UNIF}} | {{R_UNIF}} | {{G_UNIF}} |

**Observation.** {{BRANCH_OBS}}

## 7.7 Ablation: LoRA Rank

**Table 7.7: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
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

## 7.9 Peak VRAM

**Table 7.9: Peak VRAM (MB).**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 183 | {{D_VRAM}} | {{D_VRAM_PCT}}% |
| RoBERTa | 712 | {{R_VRAM}} | {{R_VRAM_PCT}}% |
| GPT-2 | 793 | {{G_VRAM}} | {{G_VRAM_PCT}}% |

## 7.10 Failure Cases

| Model | Failure | Reason |
|---|---|---|
| {{FAIL_M1}} | {{FAIL_1}} | {{FAIL_R1}} |
| {{FAIL_M2}} | {{FAIL_2}} | {{FAIL_R2}} |

## 7.11 Summary

- **Accuracy:** {{D_DELTA}}% to {{G_DELTA}}% drop.
- **Parameters:** +{{D_PARAM_PCT}}% (increase).
- **Speed:** Faster than baseline at $L \geq {{BREAKEVEN}}$; slower at $L = 128$.
- **VRAM:** Similar or slightly higher.

---
