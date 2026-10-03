<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated ($L = 128$). Primal Attention's advantage grows at 
longer $L$; not tested at $L \ge 2048$.

## 10.2 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H = 32+$)
- Mixture-of-Experts
- Vision transformers

## 10.3 Approximate Causality

Causal masking is applied to the approximate attention matrix. This is 
not equivalent to true causal attention. We do not measure the impact on 
autoregressive generation.

## 10.4 No True Speedup at Short $L$

At $L = 128$, Primal Attention is 1.3× slower. The overhead of 
computing three factors ($F_1, F_2, F_3$) plus pseudoinverse exceeds the 
savings from reduced attention matrix.

## 10.5 Segment Pooling Bias

Segment pooling is content-agnostic. Attention sinks and heavy hitters are 
not explicitly preserved.

## 10.6 No Parameter Reduction

Primal Attention does **not** reduce parameter count. For storage savings, 
use compression methods (A3, AASVD).

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Performer / Linformer

We do not directly compare with other linear attention methods. This limits 
external validity.

## 10.9 Failure Modes

Causality approximation, long L scalability

## 10.10 Code Quality

Research-grade implementation. Some numerical stability issues may remain 
at extreme $L$ or $m$.

## 10.11 Ethical Considerations

Dual-use. See Appendix E.

---
