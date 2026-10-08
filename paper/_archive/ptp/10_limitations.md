<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Score Network Sees Future Tokens

The MLP scores all tokens at once, including future positions. This is 
**acceptable for classification** but problematic for autoregressive 
generation, where future tokens are not available at inference.

**Mitigation.** For generation, use a causal score network (which we do 
not implement).

## 10.2 Shared Selection Across Queries

All queries attend to the **same** retained tokens. Per-query selection 
would be more expressive but requires per-query top-$k$.

## 10.3 Task Coverage

Only SST-2 evaluated. Long-context tasks not tested.

## 10.4 Model Coverage

Not tested on:
- Large models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.5 Very Short Sequences

At $L < 32$, pruning removes useful tokens and hurts accuracy.

## 10.6 Fixed Keep Ratio

$\rho$ is a hyperparameter, not learned. Our Adaptive Sparse Attention 
(companion method) learns a similar quantity.

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- PoWER-BERT (original)
- TR-BERT (RL)
- DynamicViT (vision)

## 10.9 Failure Modes

0.85

## 10.10 Code Quality

Research-grade implementation.

## 10.11 Ethical Considerations

Dual-use. See Appendix E.

---