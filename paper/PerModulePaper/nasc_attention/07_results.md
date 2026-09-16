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
| DistilBERT | full | 90.71% | — | 0.9099 | Dense | 1.9 s | PASS |
| DistilBERT | **NASC** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | {{D_PATH}} | {{D_TIME}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | Dense | 4.1 s | PASS |
| RoBERTa | **NASC** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | {{R_PATH}} | {{R_TIME}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | Dense | 4.2 s | PASS |
| GPT-2 | **NASC** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | Cascade | {{G_TIME}} | {{G_STATUS}} |

**Key observations:**

1. DistilBERT: {{D_OBS}}
2. RoBERTa: {{R_OBS}}
3. GPT-2: {{G_OBS}}

## 7.2 Gate Distribution

**Table 7.2: Average gate weights per branch.**

| Model | $g_0$ (local) | $g_1$ (sparse) | $g_2$ (global) |
|---|---|---|---|
| DistilBERT | N/A (dense) | N/A | N/A |
| RoBERTa | N/A (dense) | N/A | N/A |
| GPT-2 | {{G_G0}} | {{G_G1}} | {{G_G2}} |

**Observation.** {{GATE_OBS}}

## 7.3 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | Baseline | NASC | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**NASC adds:** RFF $\Omega$ ($dF$ params, frozen), gate $W_g$ ($3d$), LoRA.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Time (base) | Time (NASC) | Ratio | VRAM (base) | VRAM (NASC) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.5 Ablation: `enable_cascade_on_encoders`

**Table 7.5: Effect of cascade on encoders.**

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| True | {{D_CASCADE_ON}} | {{R_CASCADE_ON}} |
| False | {{D_CASCADE_OFF}} | {{R_CASCADE_OFF}} |

**Observation.** {{CASCADE_ENC_OBS}}

## 7.6 Ablation: Window Size

**Table 7.6: Effect of window size.**

| Window | GPT-2 acc | Time |
|---|---|---|
| 16 | {{G_W16}} | {{G_W16_T}} |
| 32 | {{G_W32}} | {{G_W32_T}} |
| 64 | {{G_W64}} | {{G_W64_T}} |
| 128 | {{G_W128}} | {{G_W128_T}} |

## 7.7 Ablation: Top-k Blocks

**Table 7.7: Effect of top-k blocks.**

| k | GPT-2 acc | Time |
|---|---|---|
| 1 | {{G_K1}} | {{G_K1_T}} |
| 2 | {{G_K2}} | {{G_K2_T}} |
| 3 | {{G_K3}} | {{G_K3_T}} |
| 4 | {{G_K4}} | {{G_K4_T}} |

## 7.8 Ablation: RFF Features

**Table 7.8: Effect of RFF features.**

| Features | GPT-2 acc | Time |
|---|---|---|
| 32 | {{G_F32}} | {{G_F32_T}} |
| 64 | {{G_F64}} | {{G_F64_T}} |
| 128 | {{G_F128}} | {{G_F128_T}} |

## 7.9 Ablation: LoRA Rank

**Table 7.9: Effect of LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.10 Comparison to Prior Modules

**Table 7.10: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| cosformer | {{COS_D}} | {{COS_R}} | {{COS_G}} | +5.7% |
| fasa | {{FASA_D}} | {{FASA_R}} | {{FASA_G}} | +2.6% |
| a3 | {{A3_D}} | {{A3_R}} | {{A3_G}} | **-60%** |
| **nasc** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.11 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: `{{D_START_LOSS}}` → `{{D_END_LOSS}}`
- RoBERTa: `{{R_START_LOSS}}` → `{{R_END_LOSS}}`
- GPT-2: `{{G_START_LOSS}}` → `{{G_END_LOSS}}`

## 7.12 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}}) — {{D_VERDICT}}
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}}) — {{R_VERDICT}}
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}}) — {{G_VERDICT}}
```

---