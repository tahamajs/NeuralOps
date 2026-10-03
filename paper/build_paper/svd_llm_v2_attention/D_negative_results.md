<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Rank

Fixed rank $r = 64$ vs energy-threshold. Accuracy: 84.8 vs 81.88.

### D.1.2 Full-Rank Factorization

No parameter reduction. Useless.

### D.1.3 Cross-Attention Support

Complex; deferred.

### D.1.4 No LoRA Recovery

Accuracy dropped to 86.4\%%.

### D.1.5 Max Pooling for GQA

We tested max pooling instead of mean. Result: 0.85, worse than mean 
0.85.

## D.2 Abandoned Variants

### D.2.1 Per-Singular-Value Quantization

We tested quantizing $\Sigma$ to INT8. Result: no significant parameter 
savings (Σ is only $r$ values), but the infrastructure is in place for 
future work.

### D.2.2 Per-Head Rank

No improvement; complexity increased.

### D.2.3 Low-Rank Bias

Negligible savings.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Energy

| $\eta$ | DistilBERT | Note |
|---|---|---|
| 0.90 | 0.85 | Too aggressive |
| 0.95 | 0.85 | Sweet spot |
| 0.99 | 0.85 | Marginal gain |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| LayerNorm recalibration | Distribution shift | <0.3% gain |
| Weight tying Q/K | Parameter sharing | Accuracy drop |
| Adaptive β in LoRA | Training stability | No change |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Comparison with Companion Methods

| Method | DistilBERT | RoBERTa | Note |
|---|---|---|---|
| MinimalSVD | 0.85 | 32 | Symmetric |
| MinimalPRISM | 0.85 | 32 | Symmetric + pre-test |
| SVD-LLM V2 | 81.88 | 50.92 | Explicit + pre-test |

## D.7 Lessons Learned

1. **Explicit storage is viable.**
2. **Energy-adaptive rank is sufficient.**
3. **LoRA recovery is essential.**
4. **GQA mean pooling works.**
5. **Pre-test prevents catastrophic failures.**

---