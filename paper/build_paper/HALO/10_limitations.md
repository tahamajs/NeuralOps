<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Speedup Without Hard Routing

HALO-ReGLA computes **both** dense and linear streams. Without a post-training 
hard routing step, no speedup is realized. This is the main limitation.

## 10.2 No Parameter Reduction

HALO-ReGLA adds gate parameters ($dH + H \approx 9K$). Total parameter count 
increases slightly.

## 10.3 Task Coverage

Only SST-2 evaluated. Long-context tasks not tested.

## 10.4 Model Coverage

Not tested on:
- Large models ($H > 32$)
- Mixture-of-Experts
- Vision transformers

## 10.5 Cumsum Memory

The causal cumsum allocates an $\mathcal{O}(L d_h^2)$ intermediate. For 
$L = 8192$, $d_h = 64$: ~$130$ MB per head, ~1.6 GB per layer for 12 heads. 
This is prohibitive.

**Future work:** Chunked cumsum or recurrent state passing.

## 10.6 Fixed Kernel

We use a fixed $\phi = \mathrm{elu} + 1$. Learned kernels could be better.

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- FLASH
- GLA
- H-Transformer

## 10.9 Failure Modes

Long-context cumsum memory

## 10.10 Code Quality

Research-grade. Numerical stability issues at extreme $L$.

## 10.11 Ethical Considerations

Dual-use. See Appendix E.

---
