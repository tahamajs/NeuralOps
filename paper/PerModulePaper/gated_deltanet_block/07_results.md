<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 (Path A: use_full_attention=True).**

| Model | Module | Accuracy | Δ | Params Δ | Time | Status |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | — | 1.9 s | PASS |
| DistilBERT | **Block (MHA)** | **{{D_MHA_ACC}}** | **{{D_MHA_DELTA}}** | +{{D_MHA_PARAM}} | {{D_MHA_TIME}} | {{D_MHA_STATUS}} |
| RoBERTa | full | 78.56% | — | — | 4.1 s | PASS |
| RoBERTa | **Block (MHA)** | **{{R_MHA_ACC}}** | {{R_MHA_DELTA}} | +{{R_MHA_PARAM}} | {{R_MHA_TIME}} | {{R_MHA_STATUS}} |
| GPT-2 | full | 79.36% | — | — | 4.2 s | PASS |
| GPT-2 | **Block (MHA)** | **{{G_MHA_ACC}}** | {{G_MHA_DELTA}} | +{{G_MHA_PARAM}} | {{G_MHA_TIME}} | {{G_MHA_STATUS}} |

**Table 7.2: Accuracy on SST-2 (Path B: use_full_attention=False).**

| Model | Module | Accuracy | Δ | Params Δ | Time | Status |
|---|---|---|---|---|---|---|
| DistilBERT | **Block (DeltaNet)** | **{{D_ACC}}** | **{{D_DELTA}}** | +{{D_PARAM}} | **{{D_STEP_TIME}} s** | {{D_STATUS}} |
| RoBERTa | **Block (DeltaNet)** | — | — | — | — | **OOM** |
| GPT-2 | **Block (DeltaNet)** | — | — | — | — | **OOM** |

## 7.2 Parameter Analysis

**Table 7.3: Parameter counts.**

| Model | Baseline | Block (MHA) | Block (DeltaNet) |
|---|---|---|---|
| DistilBERT | 66.9M | +{{D_MHA_PARAM}} | +{{D_PARAM}} |
| RoBERTa | 124.6M | +{{R_MHA_PARAM}} | +{{R_PARAM}} |
| GPT-2 | 124.4M | +{{G_MHA_PARAM}} | +{{G_PARAM}} |

**Note.** LayerNorm adds $2d$ params per block (negligible). The larger 
increase comes from the underlying attention.

## 7.3 Path A vs Path B Comparison

**Table 7.4: Path comparison on DistilBERT.**

| Path | Accuracy | Δ | Per-step time | Peak VRAM |
|---|---|---|---|---|
| Baseline | 90.71% | — | 0.4 s | 183 MB |
| Block (MHA) | {{D_MHA_ACC}}% | {{D_MHA_DELTA}} | {{D_MHA_TIME}} s | {{D_MHA_VRAM}} MB |
| Block (DeltaNet) | {{D_ACC}}% | {{D_DELTA}} | {{D_STEP_TIME}} s | {{D_VRAM}} MB |

## 7.4 Weight Transfer Verification

**Table 7.5: Weight transfer status per model (Path A).**

| Model | Q/K/V concat OK? | Out_proj OK? | Silent failure? |
|---|---|---|---|
| DistilBERT | {{D_QKV_OK}} | {{D_O_OK}} | {{D_SILENT}} |
| RoBERTa | {{R_QKV_OK}} | {{R_O_OK}} | {{R_SILENT}} |
| GPT-2 | {{G_QKV_OK}} | {{G_O_OK}} | {{G_SILENT}} |

## 7.5 LayerNorm Analysis

**Table 7.6: LayerNorm parameters after recovery (DistilBERT).**

| Metric | Value |
|---|---|
| Initial γ | 1.0 |
| Final γ mean | {{D_GAMMA_MEAN}} |
| Final γ std | {{D_GAMMA_STD}} |
| Initial β | 0.0 |
| Final β mean | {{D_BETA_MEAN}} |
| Final β std | {{D_BETA_STD}} |

**Observation.** {{LN_OBS}}

## 7.6 Ablation: Recovery Steps

**Table 7.7: Accuracy vs recovery steps (Block MHA, DistilBERT).**

| Steps | Accuracy |
|---|---|
| 10 | {{D_MHA_S10}} |
| 25 | {{D_MHA_S25}} |
| 50 | {{D_MHA_S50}} |
| 100 | {{D_MHA_S100}} |

## 7.7 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank (Block MHA, DistilBERT).**

| Rank | Accuracy |
|---|---|
| 4 | {{D_MHA_R4}} |
| 8 | {{D_MHA_R8}} |
| 16 | {{D_MHA_R16}} |
| 32 | {{D_MHA_R32}} |

## 7.8 Comparison to Attention-Level GatedDeltaNet

**Table 7.9: Block vs attention-level.**

| Module | DistilBERT | Time/step |
|---|---|---|
| Baseline | 90.71% | 0.4 s |
| GatedDeltaNet (attention-level) | {{GDN_ACC}} | {{GDN_TIME}} s |
| **GatedDeltaNetBlock** | **{{D_ACC}}** | **{{D_STEP_TIME}} s** |

**Observation.** {{BLOCK_VS_ATTN_OBS}}

## 7.9 Loss Curves

**Figure 7.1** shows recovery loss for Path A.

```
Step  Loss
1     {{D_MHA_S1}}
5     {{D_MHA_S5}}
10    {{D_MHA_S10}}
...
50    {{D_MHA_S50}}
```

## 7.10 Summary

- Path A (MHA): {{D_MHA_ACC}}% on DistilBERT
- Path B (DeltaNet): {{D_ACC}}% on DistilBERT, OOM on larger models
- LayerNorm adds +{{LN_PARAM}} params per block
- Silent weight transfer failures: {{SILENT_COUNT}} cases

**Verdict:** {{VERDICT}}

---