<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only SST-2 is evaluated. ASVD may behave differently on:
- Generation
- Multi-class classification
- Token-level tasks
- Long-context reasoning

## 10.2 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M parameters | Tested |
| >1B parameters | Not tested |
| >24 layers | Not tested |
| MoE | Not tested |
| Vision transformers | Not tested |

## 10.3 Hardware

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- CPU
- Edge devices

## 10.4 Cross-Attention Not Supported

ASVD raises `NotImplementedError` for cross-attention. This excludes 
encoder-decoder models (T5, BART).

## 10.5 `raise` on Failure

ASVD raises `RuntimeError` on:
- Missing projections
- Reconstruction error > τ

Unlike A3/AASVD, which fallback per-layer, ASVD aborts injection entirely. 
This is safer but less flexible.

## 10.6 Diagonal Weighting Only

ASVD uses diagonal weighting $\sqrt{F}$, ignoring inter-feature 
correlations. AASVD uses full covariance, which may capture more 
information.

## 10.7 No LayerNorm Recalibration

Unlike A3, ASVD does not recalibrate LayerNorms after compression. This 
may leave a small distributional shift.

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.9 No Theoretical Bound on Downstream Accuracy

We provide a reconstruction bound (Proposition 4.9.2) but no bound on 
downstream task error.

## 10.10 Activation Weighting Optional

ASVD's activation-aware weighting is optional. Without it, ASVD reduces 
to standard SVD. We report both settings.

## 10.11 Unresolved Failures

None

## 10.12 Code Quality

Research-grade implementation. Known issues:
- `raise` in `copy_weights` may be too aggressive for unusual sources
- No unit tests for edge cases
- `output_scale` parameter added but not used

---