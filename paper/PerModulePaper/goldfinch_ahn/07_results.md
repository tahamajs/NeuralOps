<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time | Status |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s | PASS |
| DistilBERT | **GoldFinch** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_TIME}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s | PASS |
| RoBERTa | **GoldFinch** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_TIME}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s | PASS |
| GPT-2 | **GoldFinch** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_TIME}} | {{G_STATUS}} |

**Key observations:**

1. {{OBS_1}}
2. {{OBS_2}}
3. {{OBS_3}}

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | GoldFinch | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Parameter breakdown per layer (DistilBERT):**
- Long-memory linear: 590,592
- Gate: 9,228
- LoRA (4 proj): 98,304
- Output scale + gate: 2
- **Total: ~698,126**

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Window Size

**Table 7.4: Effect of window size (DistilBERT).**

| Window | Acc | Time | Peak VRAM |
|---|---|---|---|
| 16 | {{D_W16}} | {{D_W16_T}} | {{D_W16_V}} |
| 32 | {{D_W32}} | {{D_W32_T}} | {{D_W32_V}} |
| 64 | {{D_W64}} | {{D_W64_T}} | {{D_W64_V}} |
| 128 | {{D_W128}} | {{D_W128_T}} | {{D_W128_V}} |
| ∞ (full) | {{D_WINF}} | {{D_WINF_T}} | {{D_WINF_V}} |

**Observation.** {{WINDOW_OBS}}

## 7.5 Ablation: Gate Init Bias

**Table 7.5: Effect of `gate_init_bias`.**

| Bias | Initial gate | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | {{D_BIAS0}} |
| 1.0 | 0.73 | {{D_BIAS1}} |
| 1.8 | 0.86 | {{D_BIAS18}} |
| 3.0 | 0.95 | {{D_BIAS3}} |

## 7.6 Ablation: Preservation Gate Init

**Table 7.6: Effect of `output_gate` init.**

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | {{D_AG0}} |
| 1.0 | 0.73 | {{D_AG1}} |
| 2.0 | 0.88 | {{D_AG2}} |
| 5.0 | 0.99 | {{D_AG5}} |

**Observation.** {{AG_OBS}}

## 7.7 Ablation: Remove Long-Memory Branch

**Table 7.7: Effect of removing the long-memory branch.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Full | {{D_ACC}}% | {{R_ACC}}% | {{G_ACC}}% |
| Without long-memory | {{D_NOLONG}}% | {{R_NOLONG}}% | {{G_NOLONG}}% |
| Δ | {{D_NOLONG_D}} | {{R_NOLONG_D}} | {{G_NOLONG_D}} |

**Observation.** {{NOLONG_OBS}}

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.9 Comparison to Prior Modules

**Table 7.9: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | {{COS_D}} | {{COS_R}} | {{COS_G}} | +5.7% |
| nha | 90.25% | OOM | 78.78% | +0.88% |
| **goldfinch_ahn** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.10 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.11 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}}) — {{D_VERDICT}}
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}}) — {{R_VERDICT}}
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}}) — {{G_VERDICT}}

---