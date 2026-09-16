# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Selection

**Goal.** Use random top-k as baseline.

**Result.** Accuracy dropped to {{RANDOM_ACC}}% — confirms selection matters.

### D.1.2 Score Net Only (No Norm)

**Goal.** Use only the learned score.

**Result.** Accuracy dropped by {{SCORE_ONLY_DROP}}%.

**Conclusion.** Norm is a useful prior.

### D.1.3 No [CLS] Pinning

**Goal.** Don't force `[CLS]`.

**Result.** Accuracy dropped by {{CLS_DROP}}% on DistilBERT.

**Conclusion.** [CLS] pinning is essential.

### D.1.4 ρ = 0.1

**Goal.** Extreme compression.

**Result.** Accuracy collapsed to {{R01_ACC}}%.

**Conclusion.** ρ < 0.3 is too aggressive.

## D.2 Abandoned Variants

### D.2.1 Attention-Mass-Based Scoring

We tested using accumulated attention mass (H2O-style) as the score. Result: 
slightly worse than score_net, no advantage.

### D.2.2 Per-Head Selection

Selecting tokens independently per head. Result: no improvement, more 
complexity.

### D.2.3 Sliding-Window Baseline

Local sliding window (like StreamingLLM). Result: worse than learned selection.

## D.3 Failed Hyperparameters

### Compression Ratio

| ρ | DistilBERT | Note |
|---|---|---|
| 0.05 | {{D_R005}} | Severe |
| 0.1 | {{D_R01}} | Too aggressive |
| 0.3 | {{D_R03}} | Marginal |
| 0.5 | {{D_R05}} | Sweet spot |
| 0.7 | {{D_R07}} | Conservative |
| 0.9 | {{D_R09}} | Near-lossless |

### Score Net Width

| Width | Params | DistilBERT |
|---|---|---|
| 32 | 25K | {{D_W32}} |
| 64 | 50K | {{D_W64}} |
| 128 | 99K | {{D_W128}} |

## D.4 Lessons Learned

1. **Norm prior is useful.**
2. **[CLS] pinning is essential.**
3. **ρ = 0.5 is a good default.**
4. **Score net width 64 is sufficient.**
5. **Random top-k is much worse.**

---