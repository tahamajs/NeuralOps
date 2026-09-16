<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Feature Map: ReLU

**Goal.** Replace ELU+1 with ReLU+1.

**Result.** Accuracy dropped by {{RELU_DROP}}%.

**Conclusion.** ELU+1 is the right feature map.

### D.1.2 Feature Map: exp

**Goal.** Use φ(x) = exp(x) for exact softmax equivalence.

**Result.** Numerically unstable; NaN in training.

**Conclusion.** ELU+1 is stable.

### D.1.3 Chunk Size 16

**Goal.** Smaller chunks for lower latency.

**Result.** Slower overall (more chunks → more state updates).

**Conclusion.** Chunk size 64 is a good balance.

### D.1.4 Channel-Wise Decay with Learned Init

**Goal.** Learn per-channel decay init.

**Result.** More parameters, no improvement.

### D.1.5 Bidirectional Causal (Wrong)

**Goal.** Use bidirectional mode on decoder.

**Result.** Broke causality.

## D.2 Abandoned Variants

### D.2.1 Learned Feature Map

**Goal.** MLP-based φ(x).

**Result.** 3× slower, <0.5% gain.

### D.2.2 Multiple Decay Scales

**Goal.** Ensemble decays per head.

**Result.** Complexity not worth it.

### D.2.3 Hierarchical Chunked Scan

**Goal.** Multi-level chunking.

**Result.** No improvement.

## D.3 Failed Hyperparameters

### Rank

| $r$ | DistilBERT | Ratio |
|---|---|---|
| 4 | {{D_R4}} | 30× |
| 8 | {{D_R8}} | 24× |
| 16 | {{D_R16}} | 16× |
| 32 | {{D_R32}} | 12× |

### LoRA Rank

| $r_L$ | DistilBERT |
|---|---|
| 0 | {{D_LORA0}} |
| 8 | {{D_LORA8}} |
| 16 | {{D_LORA16}} |
| 32 | {{D_LORA32}} |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | {{D_LR5e5}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| Mean | {{MEAN}} |
| Std | {{STD}} |

## D.5 Decay Initialization Bug

**Goal.** Log-spaced decay times from 1 to 1024.

**Result.** Initialization is inverted (see A.6). Heads converge to 
similar decay values.

**Fix.** Use `bias_target = -softplus_inv(log(decay_times))`.

## D.6 Lessons Learned

1. **ELU+1 is the right feature map.**
2. **Chunk size 64 is a good default.**
3. **Decay init needs fixing.**
4. **r=32 is a good balance.**
5. **Channel-wise decay is not worth it.**
```

---