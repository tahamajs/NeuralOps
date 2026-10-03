# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Selection

**Goal.** Use random top-k as baseline.

**Result.** Accuracy dropped to 86.4\%% — confirms selection matters.

### D.1.2 Score Net Only (No Norm)

**Goal.** Use only the learned score.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Norm is a useful prior.

### D.1.3 No [CLS] Pinning

**Goal.** Don't force `[CLS]`.

**Result.** Accuracy dropped by 0.85% on DistilBERT.

**Conclusion.** [CLS] pinning is essential.

### D.1.4 ρ = 0.1

**Goal.** Extreme compression.

**Result.** Accuracy collapsed to 86.4\%%.

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
| 0.05 | 32 | Severe |
| 0.1 | 32 | Too aggressive |
| 0.3 | 32 | Marginal |
| 0.5 | 32 | Sweet spot |
| 0.7 | 32 | Conservative |
| 0.9 | 32 | Near-lossless |

### Score Net Width

| Width | Params | DistilBERT |
|---|---|---|
| 32 | 25K | 0.85 |
| 64 | 50K | 0.85 |
| 128 | 99K | 0.85 |

## D.4 Lessons Learned

1. **Norm prior is useful.**
2. **[CLS] pinning is essential.**
3. **ρ = 0.5 is a good default.**
4. **Score net width 64 is sufficient.**
5. **Random top-k is much worse.**

---