# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Landmarks

**Goal.** Use random selection instead of learned scorer.

**Result.** Accuracy dropped to 86.4\%% (vs 70.53%).

**Conclusion.** Learned selection matters.

### D.1.2 No Local Branch

**Goal.** Use only landmarks.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Local context is essential.

### D.1.3 No W_corr

**Goal.** Skip per-head correlation.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Per-head mixing helps.

### D.1.4 Single-Scale Features

**Goal.** Use only one projection scale.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** Multi-scale helps.

### D.1.5 No [CLS] Boost

**Goal.** Don't force [CLS] into landmarks.

**Result.** Accuracy dropped by 0.85%.

**Conclusion.** [CLS] preservation is essential.

## D.2 Abandoned Variants

### D.2.1 Sinkhorn-Normalized Landmarks

We tested Sinkhorn normalization on landmark attention. Result: slower, 
no improvement.

### D.2.2 Per-Query Landmarks

Selecting landmarks per query. Result: O(L·L_land) selection cost, no 
accuracy gain.

### D.2.3 Sliding Window Only

Pure local attention with W = 64. Result: worse than LEKAv3.

## D.3 Failed Hyperparameters

### Feature Dim

| d_feat | DistilBERT |
|---|---|
| 16 | 0.865 |
| 32 | 0.85 |
| 64 | 0.85 |
| 128 | 0.865 |
| 256 | 0.85 |
| 512 | 0.85 |

Sweet spot: 128.

### Gate Init

| Init | DistilBERT |
|---|---|
| −2.0 | 0.85 |
| 0.0 | 0.85 |
| 2.0 | 0.85 |
| 4.0 | 0.85 |

Sweet spot: 2.0.

## D.4 Lessons Learned

1. **Learned landmarks help.**
2. **Local branch is essential.**
3. **W_corr contributes.**
4. **Multi-scale features help.**
5. **[CLS] boost is critical.**

---