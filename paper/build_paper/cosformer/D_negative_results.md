<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 RoBERTa Instability

Our implementation shows significant drop on RoBERTa.

### D.1.1 Attempted Fixes

| Fix | Result |
|---|---|
| Lower LR (1e-4) | 71.0% |
| Lower LR (5e-5) | 73.5% |
| Higher LoRA rank (32) | 69.5% |
| Remove cos/sin reweighting | 72.0% |
| More recovery steps | 68.5% |

**Observation.** lower LR and no reweighting help; full recovery requires further tuning

## D.2 Decoder Overhead

The chunked scan adds overhead.

### D.2.1 Attempted Optimizations

| Fix | Result |
|---|---|
| Larger chunk (64) | Faster per-step but higher VRAM |
| Smaller chunk (16) | Slower |
| Fused kernel | Not implemented |
| Vectorize outer loop | Not possible with current API |

## D.3 Cos/Sin Recomputation

We recompute cos/sin per forward. Caching would help but requires knowing 
$L$ in advance.

## D.4 RoPE Extraction Fragility

The triple-nested try/except works but is fragile.

## D.5 Dead Parameter `tau`

`self.tau` is declared but not used. This is dead code.

## D.6 Parameter Increase

Cosformer adds 1-4% parameters. Not a compression method.

## D.7 Lessons Learned

1. **Cosformer is a reasonable baseline.** Not a breakthrough, but works.
2. **RoBERTa is difficult.** Nested structure + reweighting = instability.
3. **Decoder overhead matters.** Chunked scan adds 5×.
4. **Locality recovery works.** Cos/sin reweighting helps.

## D.8 Recommendations

1. **Use Cosformer on flat encoders.**
2. **Tune hyperparameters for RoBERTa.**
3. **Consider fused kernels for decoders.**
4. **Test long sequences** to see the linear advantage.

---