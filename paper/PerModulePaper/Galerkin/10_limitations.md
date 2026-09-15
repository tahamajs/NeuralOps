# 10. Limitations

## 10.1 Encoder-Only Benefit

Galerkin provides linear complexity only for **encoder** models. Decoders 
fall back to standard attention.

## 10.2 Not a Compression Method

Parameter count increases by ~{{PARAM_CHANGE}}% via LoRA.

## 10.3 Fixed Basis

The DCT basis is fixed, not learned. A data-adaptive basis might be better.

## 10.4 Smoothness Assumption

DCT assumes smoothness. Attention patterns with sharp peaks (e.g., exact 
token matching) may be poorly represented by low-rank DCT.

## 10.5 No Formal Error Bound on Downstream Accuracy

We bound the operator approximation error (Section 4.9), but not the 
downstream task accuracy.

## 10.6 Task Coverage

Only SST-2 at $N=128$. Generation, QA, long-context untested.

## 10.7 Model Coverage

Three models (66.9M–124.6M).

## 10.8 Hardware

Only RTX 4070 Laptop.

## 10.9 Rank Selection

Fixed at $r = 64$. Rank ablation shows sensitivity.

## 10.10 Numerical Stability

DCT orthonormalization uses QR with $10^{-4}$ regularization. Extreme 
sequences may still have issues.

## 10.11 Causal Cross-Attention

Not tested.

## 10.12 Code Quality

- Research-grade
- Broad exception handling in `inject_galerkin` (returns None on any failure)
- No unit tests

## 10.13 Ethical Considerations

Dual-use. See Appendix E.
