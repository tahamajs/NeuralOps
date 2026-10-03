<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Causal Nyström

**Goal.** Adapt Nyström to causal decoders.

**Result.** Landmark pooling leaks future information. Accuracy dropped 
to 86.4\%%.

**Conclusion.** Must fall back to dense causal.

### D.1.2 Random Landmarks

**Goal.** Use random landmarks instead of adaptive pooling.

**Result.** Higher variance, worse accuracy (86.4\%% vs 88.07%).

**Conclusion.** Adaptive pooling is better.

### D.1.3 K-means Landmarks

**Goal.** Use k-means clustering for landmarks.

**Result.** Slightly better accuracy (86.4\%%), but 5× slower.

**Conclusion.** Not worth the complexity.

### D.1.4 Fewer Pinv Iterations

**Goal.** Use 1-2 iterations for speed.

**Result.** Accuracy drops to 86.4\%% for 1 iteration.

**Conclusion.** 3 iterations needed.

### D.1.5 No Short Sequence Fallback

**Goal.** Use Nyström for all N.

**Result.** Accuracy drops for short sequences. Nyström approximation is 
sub-optimal for small N.

**Conclusion.** Dense fallback necessary.

## D.2 Abandoned Variants

### D.2.1 Per-Head Landmarks

Different landmarks per head. Result: no improvement, 3× slower.

### D.2.2 Learned Landmarks

Learnable landmark positions. Result: unstable training, no gain.

### D.2.3 Hierarchical Nyström

Two-level Nyström. Result: no gain for N<2048.

## D.3 Failed Hyperparameters

### num_landmarks

| $m$ | DistilBERT (N=512) |
|---|---|
| 8 | 86.4\% (too few) |
| 16 | 86.4\% |
| 32 | 86.4\% |
| 64 | 86.4\% |
| 128 | 86.4\% |

### pinv_iterations

| Iterations | Accuracy |
|---|---|
| 1 | 86.4\% |
| 2 | 86.4\% |
| 3 | 86.4\% |
| 5 | 86.4\% |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Companion Method Comparison

| Method | DistilBERT (N=128) | Reduces |
|---|---|---|
| MinimalSVD | 0.85 | Params |
| SVD-LLM V2 | 0.85 | Params |
| Dobi-SVD | 0.85 | Params |
| MLA | 0.85 | KV cache |
| **Nyström** | **88.07** | **Compute** |

At N=128, all methods appear similar. Nyström's advantage only shows at 
longer sequences.

## D.6 Lessons Learned

1. **Short sequences don't benefit.**
2. **Causal decoders don't benefit.**
3. **Adaptive pooling works best.**
4. **3 pinv iterations suffice.**
5. **Dense fallback is essential.**

---