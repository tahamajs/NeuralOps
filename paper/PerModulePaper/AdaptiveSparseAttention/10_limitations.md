<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated ($L = 128$). ASA's sparsity benefit grows with $L$; 
we do not measure accuracy at $L \ge 512$.

## 10.2 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H = 32+$)
- Mixture-of-Experts
- Vision transformers

## 10.3 No Parameter Reduction

ASA **does not** reduce parameter count. It reduces attention memory and 
compute. For parameter savings, use A3 or AASVD.

## 10.4 No Time Reduction

Chunking bounds memory but does not reduce **time** — we still compute all 
scores. To reduce time, we would need to skip score computation for pruned 
keys.

## 10.5 Approximate Causality

Causal masking is applied to each chunk. The chunk boundary may leak 
information from prior chunks (in bidirectional settings), but for causal 
decoders, we verify strict causality.

## 10.6 Sparsity Logit Floor

The floor operation in $k_h = \lfloor (1 - \rho_h) L \rfloor$ is not 
differentiable. We use a straight-through estimator, which works but is 
not theoretically justified.

## 10.7 Single Seed

All results from seed {{SEED}}.

## 10.8 No Comparison with Some Baselines

We do not directly compare with:
- Longformer
- BigBird
- Routing Transformer
- Reformer

## 10.9 Chunk Boundary Effects

For very small chunks ($C = 32$), chunk boundary effects may hurt accuracy.

## 10.10 Failure Modes

{{UNRESOLVED_FAILURES}}

## 10.11 Code Quality

Research-grade implementation.

## 10.12 Ethical Considerations

Dual-use. See Appendix E.

---