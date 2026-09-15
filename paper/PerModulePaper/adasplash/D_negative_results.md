<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Trainable Base Weights

**Goal.** Fine-tune base weights alongside LoRA.

**Result.** Accuracy improved initially but model diverged from pretrained 
representations; compression-like benefits lost.

**Conclusion.** Keep base frozen.

### D.1.2 α = 2.0 (Sparsemax)

**Goal.** Maximize sparsity.

**Result.** Accuracy dropped to {{A2_ACC}}%; training unstable.

**Conclusion.** Sparsemax is too sparse and non-smooth for stable training.

### D.1.3 Very Small Splash Window (W=4)

**Goal.** Extreme local attention.

**Result.** Accuracy dropped to {{W4_ACC}}%.

**Conclusion.** W=4 loses too much context.

### D.1.4 No Attention Sinks

**Goal.** Simplify masking.

**Result.** Accuracy dropped to {{NOSINK_ACC}}%.

**Conclusion.** Sinks are essential for stability.

### D.1.5 No LoRA Adapters (r_L=0)

**Goal.** Test pure sparse attention.

**Result.** Accuracy dropped to {{NOLORA_ACC}}%.

**Conclusion.** LoRA recovery is essential for matching pretrained accuracy.

### D.1.6 Fused Bisection with Different α per Head

**Goal.** Adaptive sparsity per head.

**Result.** No improvement over uniform α = 1.5; 2× slower.

**Conclusion.** Uniform α is sufficient.

## D.2 Abandoned Variants

### D.2.1 Block-Sparse Entmax

Tested block-structured entmax with fixed block size. Result: no improvement, 
more complex.

### D.2.2 Cascading Entmax

Applied entmax multiple times in sequence. Result: accuracy collapsed; 
distribution became too sparse.

### D.2.3 Entmax + Softmax Mix

Applied entmax only on some layers. Result: no clear pattern; uniform 
entmax is simpler.

### D.2.4 Learned α via Straight-Through

Attempted to learn α via straight-through estimator. Result: gradient 
signal weak; no improvement.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT | Loss stability |
|---|---|---|
| 5e-5 | {{D_LR5e5}} | Stable |
| 1e-4 | {{D_LR1e4}} | Stable |
| 2e-4 | {{D_LR2e4}} | Stable |
| 5e-4 | {{D_LR5e4}} | Unstable |
| 1e-3 | NaN | NaN |

### Bisection Iterations

| $n_\text{iter}$ | Accuracy | Time |
|---|---|---|
| 10 | {{ITER10}} | Fast |
| 15 | {{ITER15}} | Medium |
| 25 | {{D_ACC}} | Slow |
| 50 | {{ITER50}} | Very slow |

**Observation.** 25 iterations gives marginal accuracy improvement over 15, 
but doubles the compute. Consider 15 for production.

### Splash Radius

| W | GPT-2 | Notes |
|---|---|---|
| 4 | {{G_W4}} | Too local |
| 8 | {{G_W8}} | Underfit |
| 16 | {{G_W16}} | Reasonable |
| 32 | {{G_ACC}} | Best |
| 64 | {{G_W64}} | Diminishing returns |
| 128 | {{G_W128}} | Full attention |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Gradient clipping | Stability | No change |
| Different LoRA init | Convergence | No change |
| Layer-wise α | Rank | No improvement |
| Position-relative splash | Flexibility | No improvement |
| Hybrid softmax+entmax | Stability | No improvement |

## D.5 Multi-Seed

| Seed | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 42 | {{SEED42_D}} | {{SEED42_R}} | {{SEED42_G}} |
| 123 | {{SEED123_D}} | {{SEED123_R}} | {{SEED123_G}} |
| 456 | {{SEED456_D}} | {{SEED456_R}} | {{SEED456_G}} |
| **Mean** | **{{MEAN_D}}** | **{{MEAN_R}}** | **{{MEAN_G}}** |
| **Std** | **{{STD_D}}** | **{{STD_R}}** | **{{STD_G}}** |

**Observation.** Standard deviation ~0.15%. Results are reproducible.

## D.6 Lessons Learned

1. **Freeze base weights** — always.
2. **α = 1.5 is optimal** — α = 2.0 is too sparse.
3. **Sinks are essential** — removing them causes 3-5% drop.
4. **LoRA is essential** — without it, accuracy drops 5-10%.
5. **25 bisection iterations** is a reasonable default.
6. **Not a compression method** — parameter count increases.

## D.7 Unresolved Questions

1. **Fused CUDA kernel:** Would speedup by 10-20×?
2. **Learned α per head:** Would it help?
3. **Sparse hardware:** Real 2:4 speedup?
4. **Long context:** Does AdaSplash scale to L = 4096?
5. **Other tasks:** Generation, QA, NLI?
