# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Energy τ < 0.80

**Goal.** Aggressive compression.

**Result.** Accuracy collapsed to {{T80_ACC}}%.

**Conclusion.** τ ≥ 0.85 required.

### D.1.2 No Residual Adapter

**Goal.** Pure SVD without adapter.

**Result.** Accuracy dropped by {{NO_ADAPTER_DROP}}%.

**Conclusion.** Residual adapter recovers significant accuracy.

### D.1.3 Asymmetric Split

**Goal.** Use Σ entirely in up.

**Result.** Slightly worse numerical stability in FP16.

**Conclusion.** Symmetric split is better.

### D.1.4 Skip W_O

**Goal.** Only factorize Q/K/V.

**Result.** Higher attention ratio but similar accuracy.

**Conclusion.** W_O factorizes fine with adapter.

### D.1.5 Residual on Down-Projection

**Goal.** Attach adapter to V_r instead of U_r.

**Result.** No improvement; slightly worse.

**Conclusion.** Adapter after up is better.

## D.2 Abandoned Variants

### D.2.1 Per-Head Rank

We tested different rank per head. Result: no improvement.

### D.2.2 Learned τ

We tested making τ learnable. Result: unstable; converged to 0.92 anyway.

### D.2.3 Two-Stage Compression

We tested SVD then LoRA then another SVD. Result: no benefit.

## D.3 Failed Hyperparameters

### Residual Rank

| $r_a$ | DistilBERT |
|---|---|
| 0 | {{D_RA0}} |
| 4 | {{D_RA4}} |
| 8 | {{D_RA8}} |
| 16 | {{D_RA16}} |
| 32 | {{D_RA32}} |

### Min Rank

| min_rank | DistilBERT |
|---|---|
| 8 | {{D_MIN8}} |
| 16 | {{D_MIN16}} |
| 32 | {{D_MIN32}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **τ = 0.92 is a good default.**
2. **Residual adapter is essential.**
3. **Symmetric split is preferable.**
4. **All four projections can be factorized.**
5. **W_O needs a slightly higher rank than Q/K.**

---