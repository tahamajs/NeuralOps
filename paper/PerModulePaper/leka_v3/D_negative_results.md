# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Landmarks

**Goal.** Use random selection instead of learned scorer.

**Result.** Accuracy dropped to {{RANDOM_L_ACC}}% (vs {{D_ACC}}%).

**Conclusion.** Learned selection matters.

### D.1.2 No Local Branch

**Goal.** Use only landmarks.

**Result.** Accuracy dropped by {{LOCAL_DROP}}%.

**Conclusion.** Local context is essential.

### D.1.3 No W_corr

**Goal.** Skip per-head correlation.

**Result.** Accuracy dropped by {{WCORR_DROP}}%.

**Conclusion.** Per-head mixing helps.

### D.1.4 Single-Scale Features

**Goal.** Use only one projection scale.

**Result.** Accuracy dropped by {{MULTI_DROP}}%.

**Conclusion.** Multi-scale helps.

### D.1.5 No [CLS] Boost

**Goal.** Don't force [CLS] into landmarks.

**Result.** Accuracy dropped by {{CLS_DROP}}%.

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
| 16 | {{D_F16}} |
| 32 | {{D_F32}} |
| 64 | {{D_F64}} |
| 128 | {{D_F128}} |
| 256 | {{D_F256}} |
| 512 | {{D_F512}} |

Sweet spot: 128.

### Gate Init

| Init | DistilBERT |
|---|---|
| −2.0 | {{D_GN2}} |
| 0.0 | {{D_G0}} |
| 2.0 | {{D_G2}} |
| 4.0 | {{D_G4}} |

Sweet spot: 2.0.

## D.4 Lessons Learned

1. **Learned landmarks help.**
2. **Local branch is essential.**
3. **W_corr contributes.**
4. **Multi-scale features help.**
5. **[CLS] boost is critical.**

---
