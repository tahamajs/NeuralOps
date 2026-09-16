<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy and timing.**

| Model | Module | Accuracy | Δ | Status | Per-step time |
|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | PASS | 1.9 s |
| DistilBERT | **GatedDeltaNet** | **{{D_ACC}}** | {{D_DELTA}} | **{{D_STATUS}}** | **{{D_STEP_TIME}} s** |
| RoBERTa | full | 78.56% | — | PASS | 4.1 s |
| RoBERTa | **GatedDeltaNet** | — | — | **{{R_STATUS}}** | — |
| GPT-2 | full | 79.36% | — | PASS | 4.2 s |
| GPT-2 | **GatedDeltaNet** | — | — | **{{G_STATUS}}** | — |

**Status legend:**
- PASS: completed successfully
- OOM: out of memory
- TIMEOUT: exceeded 3600s wall clock
- RUNNING: still executing when session ended

## 7.2 Timing Analysis

**Table 7.2: Per-step timing (5 steps).**

| Model | Baseline (ms/step) | GatedDeltaNet (ms/step) | Slowdown |
|---|---|---|---|
| DistilBERT | 400 | **{{D_STEP_MS}}** | **{{D_SLOWDOWN}}×** |
| RoBERTa | 800 | (OOM) | — |
| GPT-2 | 850 | (OOM) | — |

**Observation.** GatedDeltaNet is **~{{D_SLOWDOWN}}× slower** than baseline 
on DistilBERT, despite having lower theoretical FLOPs. This is due to Python 
loop overhead (Section 8).

## 7.3 Parameter Analysis

**Table 7.3: Parameter counts (DistilBERT).**

| Component | Baseline | GatedDeltaNet | Δ |
|---|---|---|---|
| Q projection | 589,824 | 589,824 | 0 |
| K projection | 589,824 | 589,824 | 0 |
| V projection | 589,824 | 589,824 | 0 |
| O projection | 589,824 | 589,824 | 0 |
| **Z projection** | **0** | **589,824** | **+589,824** |
| **α projection** | 0 | 9,216 | +9,216 |
| **β projection** | 0 | 9,216 | +9,216 |
| LoRA (per proj) | 0 | 24,576 × 5 | +122,880 |
| **Per layer total** | 2,359,296 | 3,539,904 | **+1,180,608** |
| **6-layer total** | 14,155,776 | 21,239,424 | **+7,083,648** |

**Δ%**: +{{D_PARAM_PCT}}% (DistilBERT)

## 7.4 Memory Analysis

**Table 7.4: Peak VRAM.**

| Model | Baseline | GatedDeltaNet |
|---|---|---|
| DistilBERT | 183 MB | **{{D_VRAM}} MB** |
| RoBERTa | 712 MB | **OOM** |
| GPT-2 | 793 MB | **OOM** |

**Observation.** Even on DistilBERT, peak VRAM increases by **{{D_VRAM_DELTA}}×**. 
On RoBERTa and GPT-2, it exceeds the 8 GB budget.

## 7.5 Ablation: Chunk Size

**Table 7.5: Effect of chunk size on per-step time (DistilBERT).**

| chunk_size | Per-step time | Peak VRAM |
|---|---|---|
| 4 | {{D_CHUNK4_TIME}} s | {{D_CHUNK4_VRAM}} MB |
| 8 | {{D_CHUNK8_TIME}} s | {{D_CHUNK8_VRAM}} MB |
| 16 | {{D_CHUNK16_TIME}} s | {{D_CHUNK16_VRAM}} MB |
| 32 | {{D_CHUNK32_TIME}} s | {{D_CHUNK32_VRAM}} MB |

**Observation.** {{CHUNK_OBS}}

## 7.6 Ablation: Gradient Checkpointing

**Table 7.6: Effect of `use_gradient_checkpointing`.**

| Setting | Per-step time | Peak VRAM | Accuracy |
|---|---|---|---|
| True | {{D_GC_TRUE_TIME}} s | {{D_GC_TRUE_VRAM}} MB | {{D_GC_TRUE_ACC}}% |
| False | {{D_GC_FALSE_TIME}} s | {{D_GC_FALSE_VRAM}} MB | {{D_GC_FALSE_ACC}}% |

**Observation.** Checkpointing trades speed for memory.

## 7.7 Ablation: Rank

**Table 7.7: Accuracy vs recovery rank.**

| LoRA rank | Per-step time | Accuracy |
|---|---|---|
| 4 | {{D_R4_TIME}} s | {{D_R4_ACC}}% |
| 8 | {{D_R8_TIME}} s | {{D_R8_ACC}}% |
| 16 | {{D_R16_TIME}} s | {{D_R16_ACC}}% |

## 7.8 Comparison to Prior Modules

**Table 7.8: Cross-module comparison (DistilBERT).**

| Module | Accuracy | Per-step time | Params Δ |
|---|---|---|---|
| full | 90.71% | 0.4 s | 0 |
| nha | 90.25% | 0.6 s | +0.88% |
| a3 | {{A3_ACC}}% | 0.5 s | **-60%** |
| **gated_delta_net** | **{{D_ACC}}%** | **{{D_STEP_TIME}} s** | **+10.7%** |

**Gated DeltaNet is the slowest module tested.**

## 7.9 Loss Curves (DistilBERT)

**Figure 7.1** shows recovery loss.

```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
10    {{D_S10}}
...
50    {{D_S50}}
```

## 7.10 Summary

- DistilBERT: **{{D_ACC}}%** after **{{D_STEP_TIME}}** s/step
- RoBERTa: **OOM** at 12 layers
- GPT-2: **OOM** at 12 layers
- Verdict: **not practical without a fused kernel**

---