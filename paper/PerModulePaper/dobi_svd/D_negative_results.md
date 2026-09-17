<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Gate Compensation

Without dividing $V$ by $\sigma(\alpha_0)$, the initial forward is scaled 
by 0.924 (for $\alpha_0 = 2.5$). This causes:
- Initial accuracy drop of 0.85%
- Slower convergence

**Conclusion.** Gate compensation is essential.

### D.1.2 Very Low Init Gate Value

$\alpha_0 = 0.0$ → $\sigma = 0.5$. Initial forward is scaled down 50%. 
Accuracy: 86.4\%%.

**Conclusion.** Use $\alpha_0 \geq 2.0$.

### D.1.3 Very High Init Gate Value

$\alpha_0 = 8.0$ → $\sigma \approx 1.0$, gradients vanish. No pruning 
observed.

**Conclusion.** Use $\alpha_0 \leq 5.0$.

### D.1.4 Zero Recovery Steps

Without recovery, gates remain at initialization. No pruning possible.

**Conclusion.** At least 30 steps needed.

### D.1.5 Aggressive Pruning

$\tau_g = 0.9$ removes too many components. Accuracy: 86.4\%%.

**Conclusion.** Use $\tau_g \in [0.3, 0.7]$.

## D.2 Abandoned Variants

### D.2.1 Hard Concrete Gates

We tested hard-concrete distribution (Louizos et al., 2018) instead of 
sigmoid. Result: no improvement, more complex.

### D.2.2 Per-Head Gates

We tested gates per attention head. Result: no improvement, complexity 
increased.

### D.2.3 Learnable Threshold

We tested making $\tau_g$ learnable. Result: threshold converges to 0.5 
empirically, no benefit from making it learnable.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 5e-5 | 0.85 |
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 (unstable) |

### Max Error

| err | Fallback % | Accuracy |
|---|---|---|
| 0.2 | 0.024% | 86.4\% |
| 0.35 | 0.024% | 86.4\% |
| 0.5 | 0.024% | 86.4\% |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal init for U/V | Slow convergence | No change |
| LayerNorm recalibration | Distribution shift | <0.3% gain |
| Separate LR for gates | Training stability | No change |
| Adaptive gate bias | Convergence | No change |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.6 Companion Method Comparison

| Method | DistilBERT | Ratio | Learned rank? |
|---|---|---|---|
| MinimalSVD | 0.85 | 2.1\times× | ❌ |
| SVD-LLM V2 | 0.85 | 2.1\times× | ❌ |
| MLA | 0.85 | 2.1\times× | ❌ |
| **Dobi-SVD** | **90.48** | **50\%×** | **✅** |

## D.7 Lessons Learned

1. **Gate compensation is essential.**
2. **Init gate value 2.0–5.0 is optimal.**
3. **At least 30 recovery steps needed.**
4. **τ_g = 0.5 is the sweet spot.**
5. **Learning rate 2e-4 works best.**

---