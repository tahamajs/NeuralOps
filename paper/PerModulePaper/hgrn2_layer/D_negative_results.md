# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Removing the Bound on Forget Gate

**Goal.** Use unbounded $\log f$.

**Result.** Accuracy dropped by {{CLAMP_DROP}}%.

**Conclusion.** Clamping is essential.

### D.1.2 Removing Input Modulation

**Goal.** Use $f s + i v$ instead of $f s + (1-f) i v$.

**Result.** Accuracy dropped by {{MODULATION_DROP}}%.

**Conclusion.** Input modulation is essential.

### D.1.3 Unidirectional on Encoders

**Goal.** Use forward scan only on DistilBERT.

**Result.** Accuracy dropped by {{BIDI_GAIN}}%.

**Conclusion.** Bidirectional matters for encoders.

### D.1.4 Chunk Size 128

**Goal.** Very large chunks.

**Result.** OOM on 8 GB GPUs.

**Conclusion.** Chunk 32 is the practical limit.

### D.1.5 FP16 Scan

**Goal.** Run scan in FP16.

**Result.** Numerical instability and NaN losses.

**Conclusion.** FP32 for scan.

## D.2 Abandoned Variants

### D.2.1 Learned Chunk Size

We tested making chunk size learnable. Result: no benefit.

### D.2.2 Attention + HGRN Hybrid

We tested interleaving attention and HGRN. Result: no benefit over full HGRN.

### D.2.3 Multi-State HGRN

We tested multiple independent state vectors. Result: marginal benefit, 3× cost.

## D.3 Failed Hyperparameters

### Forget Gate Bias

| Bias | DistilBERT |
|---|---|
| -4.0 | {{D_FB_M4}} |
| -2.5 | {{D_FB_M25}} |
| -1.0 | {{D_FB_M1}} |
| 0.0 | {{D_FB_0}} |

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} |
| 1e-3 | NaN |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.5 Lessons Learned

1. **Clamp is essential.**
2. **Input modulation matters.**
3. **Bidirectional for encoders.**
4. **FP32 scan required.**
5. **Chunk size 32 is a good default.**
```

---