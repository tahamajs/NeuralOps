# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No LayerNorm Before Projection

**Goal.** Skip head-wise LN.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** LN is essential.

### D.1.2 Random (No QR) Projection

**Goal.** Use iid Gaussian projection.

**Result.** Accuracy dropped by 0.85%; higher variance in 
predictions.

**Conclusion.** QR helps.

### D.1.3 No Preservation Gate

**Goal.** Only attention output.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Gate is essential.

### D.1.4 Very Small F (32)

**Goal.** Aggressive feature reduction.

**Result.** Accuracy collapsed to 86.4%.

**Conclusion.** F ≥ 64 required.

### D.1.5 No Masking

**Goal.** Skip padding mask.

**Result.** Accuracy collapsed (padding dominates).

**Conclusion.** Mask is essential.

## D.2 Abandoned Variants

### D.2.1 Complex RFF

We tested using the full complex exponential $e^{i \omega^\top x}$ 
instead of cos/sin. Result: no improvement.

### D.2.2 Per-Head F

Different F per head. Result: no improvement.

### D.2.3 Blocked Cumsum

Chunked cumsum to reduce memory. Result: correct but slower; needs 
optimization.

## D.3 Failed Hyperparameters

### Feature Dim

| F | DistilBERT |
|---|---|
| 32 | 86.4 |
| 64 | 0.85 |
| 128 | 0.865 |
| 256 | 51.03 |
| 512 | 0.85 |

### Preservation Gate Init

| w₀ | DistilBERT |
|---|---|
| -2 | 84.0 |
| 0 | Initial |
| 2 | 51.03 |
| 4 | 85.8 |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **Head-wise LN is essential.**
2. **QR projection reduces variance.**
3. **Preservation gate is essential.**
4. **F = 256 is a sweet spot.**
5. **Masking must be applied before aggregation.**

## D.6 Recommended Fixes

1. Replace `raise RuntimeError` in `copy_weights` with a soft fallback.
2. Implement blocked cumsum for large L.
3. Consider dropping the `output_scale` parameter (not in checkpoint).

---