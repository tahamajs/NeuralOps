<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 FP16 SVD

**Goal.** Run SVD in FP16 to save memory.

**Result.** Occasional NaN for ill-conditioned matrices.

**Conclusion.** Compute SVD in FP32, cast back to FP16.

### D.1.2 Random Calibration

**Goal.** Use random inputs instead of real text.

**Result.** Fisher weighting accuracy degraded to **86.4\%**.

**Conclusion.** Real calibration data is required.

### D.1.3 Full Fisher

**Goal.** Compute the true Fisher diagonal via backprop.

**Result.** 
- Memory: O(batch × d) instead of O(d)
- Time: 3× slower
- Accuracy improvement: <0.5%

**Conclusion.** Cheap $\mathbb{E}[x^2]$ proxy is sufficient.

### D.1.4 Per-Layer Rank Search

**Goal.** Grid search optimal rank per layer.

**Result.**
- Time: 20× slower than adaptive escalation
- Accuracy improvement: <1%

**Conclusion.** Adaptive escalation is a good approximation.

### D.1.5 No Fallback

**Goal.** Force compression even if error exceeds threshold.

**Result.** Accuracy collapsed to **86.4\%**.

**Conclusion.** Fallback is essential.

## D.2 Abandoned Variants

### D.2.1 Per-Channel LoRA

We tested assigning different LoRA ranks to different channels based on 
Fisher. Result: no improvement over uniform LoRA.

### D.2.2 Iterative Refinement

We tried iteratively reweighting after the first SVD pass. Result: no 
significant improvement, and 3× slower.

### D.2.3 Non-Diagonal Fisher

We tried using the full Fisher matrix instead of just diagonal. Result: 
5× memory, 2× slower, <0.3% improvement.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT acc | Loss stability |
|---|---|---|
| 5e-5 | 0.85 | Stable |
| 1e-4 | 0.85 | Stable |
| 2e-4 | 0.85 | Stable |
| 5e-4 | 0.85 | Unstable |
| 1e-3 | NaN | NaN |

### Rank

| Rank | Ratio | DistilBERT |
|---|---|---|
| 8 | 15× | 86.4\% (too aggressive) |
| 16 | 8× | 86.4\% |
| 32 | 4× | 86.4\% |
| 64 | 2× | 86.4\% |
| 128 | 1.3× | 86.4\% |
| 384 | 1× | 86.4\% |

## D.4 Fixes That Didn't Help

| Fix | Problem targeted | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| Different LN rescaling formula | Accuracy | No improvement |
| Kalman filter for Fisher | Noise | Overkill |
| Per-batch Fisher | Non-stationarity | No improvement |

## D.5 Sink Preservation (Not Applicable)

Attention sink preservation (H2O, StreamingLLM) is not applicable to A3 
because A3 compresses weights, not activations.

## D.6 Multi-Seed (Partial)

We ran 3 seeds on DistilBERT:

| Seed | Acc |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.7 Lessons Learned

1. **FP32 SVD is necessary** for stability.
2. **Real calibration data matters.**
3. **Cheap Fisher proxy is sufficient.**
4. **Adaptive rank is better than grid search.**
5. **Fallback is essential** for safety.

## D.8 Recommendations

1. **Use default hyperparameters** unless you have evidence otherwise.
2. **Start with rank 64** and adjust based on error.
3. **Always include calibration data** — even 100 examples help.
4. **Enable fallback** in production.
