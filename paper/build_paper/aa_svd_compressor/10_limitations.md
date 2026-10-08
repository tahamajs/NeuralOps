<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. AASVD may behave differently on:
- Generation tasks
- Multi-class classification
- Token-level tasks
- Long-context tasks

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M parameters | Tested |
| >1B parameters | Not tested |
| >24 layers | Not tested |
| Mixture-of-Experts | Not tested |
| Vision transformers | Not tested |

## 10.3 Hardware Coverage

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- CPU inference
- Edge devices (Jetson, mobile)

## 10.4 Inference Latency

Factorized layers are slower:
- Two matmuls instead of one
- Intermediate $r$-dim tensor materialized
- ~1.3× slower on GPU

## 10.5 Calibration Dependency

AASVD requires calibration. Without it:
- Covariance weighting degrades to unweighted SVD
- Accuracy drops by up to 8%

## 10.6 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.7 No Formal Bound on Downstream Accuracy

We provide a bound on **reconstruction** error (Proposition 4.8.3), but 
no bound on **downstream task** error. The gap between reconstruction and 
task accuracy is not characterized.

## 10.8 EMA Bias

EMA with β=0.9 has a bias toward recent batches. For a 5-batch calibration 
run, the first batch contributes β⁴ ≈ 0.66 weight (via $C_1 \cdot \beta^{K-1}$). 
This is a limitation for very small calibration sets.

## 10.9 Regularization Choice

Our regularization $\epsilon_\text{reg} = 10^{-6} \max \lambda + 10^{-8}$ is 
a heuristic. Different values may work better for different models.

## 10.10 No Comparison with Some Baselines

We do not directly compare with:
- DistilBERT / TinyBERT (distillation)
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)

## 10.11 High-Rank Diminishing Returns

At $r=256$, plain SVD (DRONE) matches AASVD (90.71% vs. 90.02%), suggesting the advantage of full covariance is most pronounced at low ranks where representational budget is scarce and directional energy concentration matters most. At higher ranks, the subspace capacity is sufficiently large that even unweighted truncation captures the dominant variance.

## 10.12 Code Quality

Research-grade implementation:
- Broad exception handling in some paths
- No unit tests for edge cases
- Numerical stability could be improved

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
