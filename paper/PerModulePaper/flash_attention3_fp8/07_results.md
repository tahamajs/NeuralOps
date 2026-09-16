<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Hardware Detection

**Table 7.1: Hardware detection results on RTX 4070 Laptop.**

| Check | Result |
|---|---|
| `flash_attn` importable | **{{FLASH_AVAILABLE}}** |
| `torch.float8_e4m3fn` exists | **{{FP8_DTYPE}}** |
| `q.is_cuda` at forward | True |
| FP8 path taken | **{{FP8_TAKEN}}** |
| SDPA backend | **{{SDPA_BACKEND}}** |

**Interpretation.** {{FP8_INTERPRETATION}}

## 7.2 Main Results

**Table 7.2: Accuracy on SST-2 validation.**

| Model | Module | Acc | Δ | F1 | Params Δ | Time | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | — | 1.9 s | 183 MB |
| DistilBERT | **flash3_fp8** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAM_DELTA}} | {{D_TIME}} | {{D_VRAM}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s | 712 MB |
| RoBERTa | **flash3_fp8** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAM_DELTA}} | {{R_TIME}} | {{R_VRAM}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s | 793 MB |
| GPT-2 | **flash3_fp8** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAM_DELTA}} | {{G_TIME}} | {{G_VRAM}} |

**Key observations:**

1. {{OBS_1}}
2. {{OBS_2}}
3. {{OBS_3}}

## 7.3 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | Baseline | After LoRA | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Note.** This module is **not a compression method**. Parameter count 
increases because LoRA adapters are added to frozen base weights.

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Time (baseline) | Time (ours) | Ratio | VRAM (baseline) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

**Observation.** {{EFFICIENCY_OBS}}

## 7.5 Ablation: `use_fp8 = False`

**Table 7.5: Effect of forcing SDPA path.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `use_fp8=True` | {{D_USE_TRUE}} | {{R_USE_TRUE}} | {{G_USE_TRUE}} |
| `use_fp8=False` | {{D_USE_FALSE}} | {{R_USE_FALSE}} | {{G_USE_FALSE}} |
| Δ | {{D_USE_DELTA}} | {{R_USE_DELTA}} | {{G_USE_DELTA}} |

**Observation.** On RTX 4070, both configurations take the SDPA path, so 
results should be **identical**.

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |

## 7.7 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| hamming | 79.13% | 50.92% | 71.22% | +0.88% |
| **flash3_fp8** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: {{D_LOSS_TRAJ}}
- RoBERTa: {{R_LOSS_TRAJ}}
- GPT-2: {{G_LOSS_TRAJ}}

## 7.10 Summary

- FP8 path taken: **{{FP8_TAKEN}}**
- Accuracy preservation: **{{D_DELTA}}%**, **{{R_DELTA}}%**, **{{G_DELTA}}%**
- Parameters: **increase** by **{{PARAM_PCT}}%**
- Verdict: **{{VERDICT}}**

---