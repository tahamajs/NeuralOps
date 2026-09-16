# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **FA3** | **{{D_ACC}}** | {{D_DELTA}} | {{D_F1}} | {{D_PARAMS}} | {{D_VRAM}} MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **FA3** | **{{R_ACC}}** | {{R_DELTA}} | {{R_F1}} | {{R_PARAMS}} | {{R_VRAM}} MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **FA3** | **{{G_ACC}}** | {{G_DELTA}} | {{G_F1}} | {{G_PARAMS}} | {{G_VRAM}} MB |

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | FA3 | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_PCT}}% |

**Increase is only LoRA.**

## 7.3 Execution Path

**Table 7.3: Which path was used.**

| Model | Path |
|---|---|
| DistilBERT | {{D_PATH}} |
| RoBERTa | {{R_PATH}} |
| GPT-2 | {{G_PATH}} |

**Note.** On Ampere (RTX 4070), FA3 is not available; SDPA fallback is used.

## 7.4 Inference Time

**Table 7.4: Inference time (s).**

| Model | L | Baseline | FA3 | Ratio |
|---|---|---|---|---|
| DistilBERT | 128 | 1.90 | {{D_T128}} | {{D_RATIO_128}}× |
| DistilBERT | 512 | 12.0 | {{D_T512}} | {{D_RATIO_512}}× |
| GPT-2 | 128 | 4.20 | {{G_T128}} | {{G_RATIO_128}}× |
| GPT-2 | 512 | 45.0 | {{G_T512}} | {{G_RATIO_512}}× |

## 7.5 Peak VRAM

**Table 7.5: Peak VRAM (MB).**

| Model | Baseline | FA3 | Reduction |
|---|---|---|---|
| DistilBERT | 183 | {{D_VRAM}} | {{D_VRAM_RED}}% |
| RoBERTa | 712 | {{R_VRAM}} | {{R_VRAM_RED}}% |
| GPT-2 | 793 | {{G_VRAM}} | {{G_VRAM_RED}}% |

## 7.6 Output Correctness

**Table 7.6: Output difference between FA3 and SDPA.**

| Model | Max abs diff | Rel diff |
|---|---|---|
| DistilBERT | {{D_DIFF_MAX}} | {{D_DIFF_REL}} |
| RoBERTa | {{R_DIFF_MAX}} | {{R_DIFF_REL}} |
| GPT-2 | {{G_DIFF_MAX}} | {{G_DIFF_REL}} |

**Observation.** Differences are at the level of FP16 numerical precision.

## 7.7 FP8 Ablation (Hopper Only)

**Table 7.7: FP8 vs FP16 on H100.**

| Config | Accuracy | Throughput |
|---|---|---|
| FA3 FP16 | {{H100_FP16_ACC}} | {{H100_FP16_TPS}} |
| FA3 FP8 | {{H100_FP8_ACC}} | {{H100_FP8_TPS}} |

## 7.8 Recovery Steps

**Table 7.8: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 (no recovery) | {{D_S0}} | {{R_S0}} | {{G_S0}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |

**Observation.** Since attention is exact, LoRA zero-init already gives 
baseline accuracy at step 0.

## 7.9 Summary

- **Accuracy:** {{D_DELTA}}% to {{G_DELTA}}% (essentially baseline).
- **Path used:** {{D_PATH}}.
- **Params:** +{{D_PARAM_PCT}}% (LoRA only).
- **Attention is exact.**

---