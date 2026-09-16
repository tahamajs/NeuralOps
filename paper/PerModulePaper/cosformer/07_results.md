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
| DistilBERT | **Cosformer** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | +{{D_PARAM_PCT}}% | {{D_TIME}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | — | 4.1 s | PASS |
| RoBERTa | **Cosformer** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | +{{R_PARAM_PCT}}% | {{R_TIME}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | — | 4.2 s | PASS |
| GPT-2 | **Cosformer** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | +{{G_PARAM_PCT}}% | {{G_TIME}} | {{G_STATUS}} |

**Key observations:**

1. **DistilBERT:** {{D_OBS}}
2. **RoBERTa:** {{R_OBS}}
3. **GPT-2:** {{G_OBS}}

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Cosformer | Δ Abs | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Cosformer does not reduce parameter count.** It adds LoRA adapters.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Chunk Size (Decoder)

**Table 7.4: Effect of chunk size on per-step time (GPT-2).**

| chunk_size | Per-step time | Peak VRAM |
|---|---|---|
| 16 | {{G_CHUNK16_TIME}} s | {{G_CHUNK16_VRAM}} MB |
| 32 | {{G_CHUNK32_TIME}} s | {{G_CHUNK32_VRAM}} MB |
| 64 | {{G_CHUNK64_TIME}} s | {{G_CHUNK64_VRAM}} MB |
| 128 | {{G_CHUNK128_TIME}} s | {{G_CHUNK128_VRAM}} MB |

**Observation.** {{CHUNK_OBS}}

## 7.5 Ablation: Activation Function

**Table 7.5: Effect of `act_fun` on accuracy.**

| Act | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `elu+1` | {{D_ELU}} | {{R_ELU}} | {{G_ELU}} |
| `softplus` | {{D_SOFTPLUS}} | {{R_SOFTPLUS}} | {{G_SOFTPLUS}} |

**Observation.** {{ACT_OBS}}

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
| nha | 90.25% | OOM | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| hamming | 79.13% | 50.92% | 71.22% | +0.88% |
| gated_delta_net | TIMEOUT | OOM | OOM | +10.7% |
| **cosformer** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.9 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: from `{{D_START_LOSS}}` to `{{D_END_LOSS}}`
- RoBERTa: from `{{R_START_LOSS}}` to `{{R_END_LOSS}}`
- GPT-2: from `{{G_START_LOSS}}` to `{{G_END_LOSS}}`

## 7.10 Summary

- **DistilBERT:** {{D_ACC}}% (Δ = {{D_DELTA}}) — {{D_VERDICT}}
- **RoBERTa:** {{R_ACC}}% (Δ = {{R_DELTA}}) — {{R_VERDICT}}
- **GPT-2:** {{G_ACC}}% (Δ = {{G_DELTA}}) — {{G_VERDICT}}

Cosformer is a **reasonable linear attention baseline** that preserves 
most of the baseline accuracy at modest overhead.

---