# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | Trainable | VRAM |
|---|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 0 | 183 MB |
| DistilBERT | **HGRN2Layer** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_TRAINABLE}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 0 | 712 MB |
| RoBERTa | **HGRN2Layer** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_TRAINABLE}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 0 | 793 MB |
| GPT-2 | **HGRN2Layer** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_TRAINABLE}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**We state explicitly:** HGRN2Layer increases parameter count.

## 7.3 Inference Speed vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | Ours | Speedup |
|---|---|---|---|
| 128 | 1.90 | {{D_T128}} | {{S128}}× |
| 256 | 3.50 | {{D_T256}} | {{S256}}× |
| 512 | 12.0 | {{D_T512}} | {{S512}}× |
| 1024 | 45.0 | {{D_T1024}} | {{S1024}}× |
| 2048 | 180.0 | {{D_T2048}} | {{S2048}}× |
| 4096 | 720.0 | {{D_T4096}} | {{S4096}}× |

**Breakeven:** $L^* = {{BREAKEVEN}}$.

## 7.4 Ablation: Chunk Size

**Table 7.4: Effect of chunk size.**

| Chunk | DistilBERT | Time (L=512) |
|---|---|---|
| 8 | {{D_C8}} | {{D_C8_T}} |
| 16 | {{D_C16}} | {{D_C16_T}} |
| 32 | {{D_C32}} | {{D_C32_T}} |
| 64 | {{D_C64}} | {{D_C64_T}} |

## 7.5 Ablation: Bidirectional vs Causal

**Table 7.5: Effect of scan direction.**

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Bidirectional | {{D_BIDI}} | {{R_BIDI}} | N/A |
| Causal | {{D_CAUSAL}} | {{R_CAUSAL}} | {{G_CAUSAL}} |

## 7.6 Ablation: LoRA Rank

**Table 7.6: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
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

## 7.8 Peak VRAM

**Table 7.8: Peak VRAM (MB).**

| Model | Baseline | Ours | Δ |
|---|---|---|---|
| DistilBERT | 183 | {{D_VRAM}} | {{D_VRAM_PCT}}% |
| RoBERTa | 712 | {{R_VRAM}} | {{R_VRAM_PCT}}% |
| GPT-2 | 793 | {{G_VRAM}} | {{G_VRAM_PCT}}% |

## 7.9 Training Curves

**Figure 7.1** shows loss vs step. {{TRAINING_DESC}}

## 7.10 Summary

- **Accuracy:** drops {{D_DELTA}}% to {{G_DELTA}}%.
- **Parameters:** increase {{D_PARAM_PCT}}% to {{G_PARAM_PCT}}%.
- **Speed:** slower at L=128, faster at L≥{{BREAKEVEN}}.
- **VRAM:** similar or slightly higher.

---
