<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Fixed Rank

Fixed rank $r = 64$ vs energy-threshold. Accuracy: {{FIXED64}} vs {{D_ACC}}.

### D.1.2 Full-Rank Factorization

No parameter reduction. Useless.

### D.1.3 Cross-Attention Support

Complex; deferred.

### D.1.4 No LoRA Recovery

Accuracy dropped to {{NO_LORA_ACC}}%.

### D.1.5 Max Pooling for GQA

We tested max pooling instead of mean. Result: {{MAX_POOL}}, worse than mean 
{{MEAN_POOL}}.

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
| 5e-5 | {{D_LR5e5}} |
| 1e-4 | {{D_LR1e4}} |
| 2e-4 | {{D_LR2e4}} |
| 5e-4 | {{D_LR5e4}} (unstable) |

### Energy

| $\eta$ | DistilBERT | Note |
|---|---|---|
| 0.90 | {{D_E90}} | Too aggressive |
| 0.95 | {{D_E95}} | Sweet spot |
| 0.99 | {{D_E99}} | Marginal gain |

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
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Comparison with Companion Methods

| Method | DistilBERT | RoBERTa | Note |
|---|---|---|---|
| MinimalSVD | {{MINSVD_D}} | {{MINSVD_R}} | Symmetric |
| MinimalPRISM | {{PRISM_D}} | {{PRISM_R}} | Symmetric + pre-test |
| SVD-LLM V2 | {{D_ACC}} | {{R_ACC}} | Explicit + pre-test |

## D.7 Lessons Learned

1. **Explicit storage is viable.**
2. **Energy-adaptive rank is sufficient.**
3. **LoRA recovery is essential.**
4. **GQA mean pooling works.**
5. **Pre-test prevents catastrophic failures.**

---