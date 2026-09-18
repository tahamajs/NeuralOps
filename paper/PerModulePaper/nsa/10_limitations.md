<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Time Speedup Without Sparse Kernels

NSA computes the full $L \times L$ score matrix then masks. Without sparse 
matmul kernels that skip masked entries, no wall-clock speedup is realized.

## 10.2 No Parameter Reduction

NSA adds **zero** parameters. Its benefit is attention sparsity, not 
parameter compression.

## 10.3 Fixed Hyperparameters

$w$ and $k$ are hyperparameters, not learned per head or per layer. Our 
Adaptive Sparse Attention (companion method) does learn these.

## 10.4 Task Coverage

Only SST-2 evaluated at $L = 128$. Long-context tasks ($L > 512$) not 
tested.

## 10.5 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Wide models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.6 No Speedup at Short L

At $L = 128$ with $w = 32, k = 16$: density is 63%. Small benefit.

## 10.7 Top-K Time Complexity

`torch.topk` on $L \times L$ costs $\mathcal{O}(L^2)$. NSA does not reduce 
asymptotic time complexity.

## 10.8 Single Seed

All results from seed 42.

## 10.9 No Comparison with Some Baselines

We do not compare with:
- Longformer
- BigBird
- Routing Transformer

## 10.10 Failure Modes

0.85

## 10.11 Code Quality

Research-grade implementation.

## 10.12 Ethical Considerations

Dual-use. See Appendix E.

---