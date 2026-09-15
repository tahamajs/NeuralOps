<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

This module is a **negative result**. The entire appendix is dedicated to 
documenting what failed.

## D.1 Encoder Path Fails

| Model | Baseline | Galerkin | Δ |
|---|---|---|---|
| DistilBERT | 90.71% | {{D_ACC}}% | {{D_DELTA}}% |
| RoBERTa | 78.56% | {{R_ACC}}% | {{R_DELTA}}% |

## D.2 Decoder Path is No-Op

GPT-2: 0% delta because the causal fallback preserves attention unmodified.

## D.3 Attempts to Fix

### D.3.1 Increase Rank

We tested $r \in \{8, 16, 32, 64, 128\}$:

| $r$ | DistilBERT |
|---|---|
| 8 | {{D_R8}} |
| 16 | {{D_R16}} |
| 32 | {{D_R32}} |
| 64 | {{D_R64}} |
| 128 | {{D_R128}} |

**Result.** Even at $r = 128$ (equal to sequence length), accuracy does 
not recover. **Truncation is not the problem; the basis is.**

### D.3.2 Learnable Basis

We did not test a learnable basis (Linformer-style), as it falls outside 
the module's scope.

### D.3.3 Different Orthonormalization

We tested alternative orthonormalization (Gram-Schmidt, Householder). No 
improvement.

### D.3.4 Remove Energy Correction

Without $\sqrt{N/r}$ correction:

| Config | DistilBERT |
|---|---|
| With correction | {{D_ACC}}% |
| Without | {{D_WITHOUT_CORR}}% |

**Result.** Removing the correction makes it worse.

## D.4 Failed Hyperparameters

### Basis Rank

See D.3.1.

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | {{LR_1e5}} |
| 5e-5 | {{LR_5e5}} |
| 1e-4 | {{LR_1e4}} |
| 2e-4 | {{LR_2e4}} (default) |
| 5e-4 | {{LR_5e4}} (NaN) |

### Recovery Steps

| Steps | DistilBERT | RoBERTa |
|---|---|---|
| 10 | {{D_S10}} | {{R_S10}} |
| 25 | {{D_S25}} | {{R_S25}} |
| 50 | {{D_S50}} | {{R_S50}} |
| 100 | {{D_S100}} | {{R_S100}} |

**Result.** More steps do not help.

## D.5 Hypotheses Tested

| Hypothesis | Test | Result |
|---|---|---|
| H1: Truncation loses energy | Rank sweep | Rejected |
| H2: Fixed basis cannot adapt | Attention heatmap | **Confirmed** |
| H3: Energy correction overshoots | Remove correction | Rejected |
| H4: Nested encoder mis-detected | Check `is_nested_encoder` | Rejected |

## D.6 Lessons Learned

1. **Attention is fundamentally adaptive.** Fixed bases cannot capture it.
2. **Spectral methods (FNet, Linformer) must learn or sample their basis.** 
   DCT alone is insufficient.
3. **Layer depth amplifies error.** RoBERTa (12 layers) collapses; DistilBERT 
   (6 layers) survives with heavy loss.
4. **Decoder fallback is not a solution.** It is a no-op.
5. **Rank increase does not help.** The problem is the basis, not the rank.

## D.7 Recommendations for Future Work

1. **Use learnable bases.** Linformer-style projections or Nyström landmarks.
2. **Test data-dependent DCT** (basis from each batch).
3. **Hybrid approaches:** DCT as a warm-start, then fine-tune basis.
4. **Smaller models first.** Test on TinyBERT before scaling to BERT.
5. **Measure attention sharpness** to predict whether projection will work.

## D.8 What This Module Should Not Be Used For

- Production deployment
- Long-context attention
- Parameter reduction
- Any task requiring accuracy preservation

---
