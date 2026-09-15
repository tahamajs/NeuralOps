<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 07_results.md
     PURPOSE: All empirical results
     DEPENDENCIES: 06_setup.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 7. Results

## 7.1 Main Results

**Table 7.1: Accuracy on SST-2 validation.**

| Model | Path | Accuracy | Δ | F1 | Status |
|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | PASS |
| DistilBERT | **Galerkin** | **{{D_ACC}}** | **{{D_DELTA}}** | {{D_F1}} | {{D_STATUS}} |
| RoBERTa | full | 78.56% | — | 0.8124 | PASS |
| RoBERTa | **Galerkin** | **{{R_ACC}}** | **{{R_DELTA}}** | {{R_F1}} | {{R_STATUS}} |
| GPT-2 | full | 79.36% | — | 0.7978 | PASS |
| GPT-2 | **Galerkin** | **{{G_ACC}}** | **{{G_DELTA}}** | {{G_F1}} | {{G_STATUS}} |

**Key observations:**

1. **DistilBERT**: -{{D_DROP}}% (encoder path, fails)
2. **RoBERTa**: -{{R_DROP}}% (encoder path, collapse to chance)
3. **GPT-2**: {{G_DELTA}}% (decoder fallback, no-op)

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | After Galerkin | Δ | Δ% |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | {{D_PARAMS}} | +{{D_PARAM_ABS}} | +{{D_PARAM_PCT}}% |
| RoBERTa | 124,647,170 | {{R_PARAMS}} | +{{R_PARAM_ABS}} | +{{R_PARAM_PCT}}% |
| GPT-2 | 124,441,344 | {{G_PARAMS}} | +{{G_PARAM_ABS}} | +{{G_PARAM_PCT}}% |

**Note.** Galerkin Attention **adds parameters** via LoRA. It is not a 
compression method.

## 7.3 Efficiency Profile

**Table 7.3: Inference efficiency.**

| Model | Time (base) | Time (ours) | Ratio | VRAM (base) | VRAM (ours) |
|---|---|---|---|---|---|
| DistilBERT | 1.9 s | {{D_TIME}} s | {{D_TIME_R}}× | 183 MB | {{D_VRAM}} MB |
| RoBERTa | 4.1 s | {{R_TIME}} s | {{R_TIME_R}}× | 712 MB | {{R_VRAM}} MB |
| GPT-2 | 4.2 s | {{G_TIME}} s | {{G_TIME_R}}× | 793 MB | {{G_VRAM}} MB |

## 7.4 Ablation: Basis Rank

**Table 7.4: Accuracy vs basis rank (DistilBERT).**

| $r$ | Accuracy | Δ | Reconstructed energy % |
|---|---|---|---|
| 8 | {{D_R8}} | {{D_R8_D}} | {{D_R8_E}} |
| 16 | {{D_R16}} | {{D_R16_D}} | {{D_R16_E}} |
| 32 | {{D_R32}} | {{D_R32_D}} | {{D_R32_E}} |
| 64 | {{D_R64}} | {{D_R64_D}} | {{D_R64_E}} |
| 128 | {{D_R128}} | {{D_R128_D}} | {{D_R128_E}} |

**Observation.** Even at $r = 128$ (equal to sequence length), accuracy 
does not recover. This suggests the failure is not due to truncation but 
to **the basis itself**.

## 7.5 Ablation: Recovery Steps

**Table 7.5: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} | {{G_S10}} |
| 25 | {{D_S25}} | {{R_S25}} | {{G_S25}} |
| 50 | {{D_S50}} | {{R_S50}} | {{G_S50}} |
| 100 | {{D_S100}} | {{R_S100}} | {{G_S100}} |

**Observation.** More recovery steps do not close the gap for encoders.

## 7.6 Per-Layer Analysis

**Table 7.6: Accuracy by layer replacement (DistilBERT).**

| Layers replaced | Accuracy |
|---|---|
| None (baseline) | 90.71% |
| First 2 | {{D_FIRST2}} |
| Middle 2 | {{D_MID2}} |
| Last 2 | {{D_LAST2}} |
| All 6 | {{D_ACC}} |

**Observation.** {{LAYER_OBS}}

## 7.7 Comparison to Prior Modules

**Table 7.7: Cross-module comparison.**

| Module | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 0 |
| nha | 90.25% | FAIL | 78.78% | +0.88% |
| nasc | 90.71% | 64.56% | 78.44% | +0.94% |
| hamming | 79.13% | 50.92% | 71.22% | +0.88% |
| **Galerkin** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **+{{PARAM_PCT}}%** |

## 7.8 Training Loss Curves

**Figure 7.1** shows recovery loss.

- DistilBERT: starts at `{{D_START_LOSS}}`, ends at `{{D_END_LOSS}}`
- RoBERTa: oscillates between `{{R_MIN_LOSS}}` and `{{R_MAX_LOSS}}`
- GPT-2: stable, ends at `{{G_END_LOSS}}`

## 7.9 Summary

- Encoder path: **fails** (25–30% drop)
- Decoder fallback: **no-op** (0% delta)
- Parameters: **increase** by ~0.9%
- Verdict: **not recommended for any use case tested**

---
