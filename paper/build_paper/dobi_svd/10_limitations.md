<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. Dobi-SVD may behave differently on:
- Generation tasks
- Multi-class classification
- Token-level tasks

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| >24 layers | Not tested |

## 10.3 Hardware

Only tested on RTX 4070 Laptop (8 GB).

## 10.4 Gates Do Not Always Prune

Some gates may not drift far enough from initialization to be prunable. In 
our experiments, **48.5\%%** of gates remained above τ_g = 0.7.

## 10.5 Recovery Steps Matter

Gate convergence requires **at least 30–50 recovery steps**. With fewer 
steps, gates remain near initialization and no pruning is possible.

## 10.6 Post-Hoc Pruning is Not Exact

Pruning gates below τ_g removes their contribution entirely. This is a 
hard approximation. In some cases, we observed **0.85%** accuracy 
drop after pruning.

## 10.7 No Formal Rank Convergence Guarantee

We provide no proof that gates converge to the optimal rank. Gates may 
get stuck in local minima.

## 10.8 Reconstruction Bound Not Enforced per Layer

Unlike A3, Dobi-SVD does not enforce a Frobenius threshold per layer. It 
relies on `max_error` at initialization and gate learning afterward.

## 10.9 Inference Latency

Dobi-SVD adds an elementwise gate multiplication. On GPU, this is cheap 
but not free.

## 10.10 Cross-Attention Not Supported

Encoder-decoder models (T5, BART) are not supported.

## 10.11 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.12 Heuristic Hyperparameters

`init_gate_val`, `init_rank_ratio`, `max_error`, and `τ_g` are all empirical.

## 10.13 No Comparison with Some Baselines

We do not compare with:
- Post-hoc rank search (too slow)
- Learned gate methods (AdaLoRA)

## 10.14 Code Quality

Research-grade. Known issues:
- Broad exception handling
- No unit tests for edge cases
- `raise RuntimeError` in some paths should log

## 10.15 Ethical Considerations

See Appendix E.

---