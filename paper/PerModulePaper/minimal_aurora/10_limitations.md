# 10. Limitations

## 10.1 No Activation Weighting

MinimalAURORA uses unweighted SVD. A3 and AASVD achieve similar accuracy 
with weighting but more complexity.

## 10.2 No LayerNorm Recalibration

Unlike A3, no LayerNorm rescaling is applied. The distributional shift 
after compression is not corrected.

## 10.3 `pres_gate` Unused

A `pres_gate` parameter is defined but never used in `forward`. This is 
a code smell and adds 1 parameter.

## 10.4 `output_scale` Drift

`output_scale` is trainable and drifts during recovery. If it reaches 0, 
attention output collapses.

## 10.5 Silent Injection Failure

`inject_minimal_aurora` returns `None` on failure. Callers must check 
the return value.

## 10.6 Cross-Attention Limited

Cross-attention is supported but not thoroughly tested.

## 10.7 Task Coverage

Only SST-2 is evaluated.

## 10.8 Model Coverage

Tested on ≤124M parameter models.

## 10.9 Hardware

All experiments on RTX 4070 Laptop.

## 10.10 Single-Seed

All main results from a single seed.

## 10.11 No Theoretical Bound on Downstream Accuracy

We provide a bound on reconstruction error (A.1) but no bound on 
downstream task error.

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade. Known issues:
- `pres_gate` unused
- `output_scale` drift
- Silent injection failure
- No unit tests

---