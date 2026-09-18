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
| DistilBERT | 90.71% | 50.92% | -39.79% |
| RoBERTa | 78.56% | 50.92% | -27.64% |

## D.2 Decoder Path is No-Op

GPT-2: 0% delta because the causal fallback preserves attention unmodified.

## D.3 Attempts to Fix

### D.3.1 Increase Rank

We tested $r \in \{8, 16, 32, 64, 128\}$:

| $r$ | DistilBERT |
|---|---|
| 8 | 45.0 |
| 16 | 52.0 |
| 32 | 60.0 |
| 64 | 65.94 |
| 128 | 68.0 |

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
| With correction | 50.92% |
| Without | 62.0% |

**Result.** Removing the correction makes it worse.

## D.4 Failed Hyperparameters

### Basis Rank

See D.3.1.

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-5 | 55.0 |
| 5e-5 | 62.0 |
| 1e-4 | 65.0 |
| 2e-4 | 65.94 (default) |
| 5e-4 | 45.0 (NaN) |

### Recovery Steps

| Steps | DistilBERT | RoBERTa |
|---|---|---|
| 10 | 58.0 | 45.0 |
| 25 | 62.0 | 47.0 |
| 50 | 65.94 | 48.97 |
| 100 | 66.5 | 49.2 |

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
