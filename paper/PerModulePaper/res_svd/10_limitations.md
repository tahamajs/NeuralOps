# 10. Limitations

## 10.1 Silent Injection Failure

`inject_res_svd` catches exceptions and returns `None`. If the caller 
ignores this, the model runs as baseline without warning.

## 10.2 Cross-Attention Not Supported

Raises `NotImplementedError` for `encoder_hidden_states`. Encoder-decoder 
models (T5, BART) are excluded.

## 10.3 No Attention Complexity Reduction

Res-SVD keeps softmax attention ($\mathcal{O}(N^2)$). Only projections are 
compressed.

## 10.4 Equivalence to Rank-$(r_b+r_r)$ SVD

As proven in §4.7.1, the two-level structure is mathematically equivalent 
to single-level SVD of rank $r_b + r_r$. The advantage is the PEFT 
structure, not the approximation quality.

## 10.5 Task Coverage

Only SST-2 is evaluated.

## 10.6 Model Coverage

Tested on ≤124M parameter models.

## 10.7 Hardware

All experiments on RTX 4070 Laptop.

## 10.8 Single-Seed

All main results from a single seed.

## 10.9 No Activation Weighting

Unlike A3 and AASVD, Res-SVD uses unweighted SVD. Adding activation-aware 
weighting could improve results.

## 10.10 Chunked Context

Only $N \leq 512$ tested. Long-context behavior unknown.

## 10.11 Unresolved Failures

0.85

## 10.12 Code Quality

Research-grade. Known issues:
- Silent injection fallback
- No unit tests
- `raise` in `copy_weights` may be too strict

---