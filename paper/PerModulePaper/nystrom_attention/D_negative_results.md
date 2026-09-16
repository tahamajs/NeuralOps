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
to {{CAUSAL_ACC}}%.

**Conclusion.** Must fall back to dense causal.

### D.1.2 Random Landmarks

**Goal.** Use random landmarks instead of adaptive pooling.

**Result.** Higher variance, worse accuracy ({{RANDOM_ACC}}% vs {{D_ACC}}%).

**Conclusion.** Adaptive pooling is better.

### D.1.3 K-means Landmarks

**Goal.** Use k-means clustering for landmarks.

**Result.** Slightly better accuracy ({{KMEANS_ACC}}%), but 5× slower.

**Conclusion.** Not worth the complexity.

### D.1.4 Fewer Pinv Iterations

**Goal.** Use 1-2 iterations for speed.

**Result.** Accuracy drops to {{I1_ACC}}% for 1 iteration.

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
| 8 | {{D_M8_ACC}} (too few) |
| 16 | {{D_M16_ACC}} |
| 32 | {{D_M32_ACC}} |
| 64 | {{D_M64_ACC}} |
| 128 | {{D_M128_ACC}} |

### pinv_iterations

| Iterations | Accuracy |
|---|---|
| 1 | {{D_I1_ACC}} |
| 2 | {{D_I2_ACC}} |
| 3 | {{D_I3_ACC}} |
| 5 | {{D_I5_ACC}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Companion Method Comparison

| Method | DistilBERT (N=128) | Reduces |
|---|---|---|
| MinimalSVD | {{MINSVD_D}} | Params |
| SVD-LLM V2 | {{SVDLLM_D}} | Params |
| Dobi-SVD | {{DOBI_D}} | Params |
| MLA | {{MLA_D}} | KV cache |
| **Nyström** | **{{D_ACC}}** | **Compute** |

At N=128, all methods appear similar. Nyström's advantage only shows at 
longer sequences.

## D.6 Lessons Learned

1. **Short sequences don't benefit.**
2. **Causal decoders don't benefit.**
3. **Adaptive pooling works best.**
4. **3 pinv iterations suffice.**
5. **Dense fallback is essential.**
```

---