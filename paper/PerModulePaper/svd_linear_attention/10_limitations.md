<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Approximation Error

SVD-Linear uses the ELU+1 feature map, which approximates but does not equal 
softmax attention. For tasks requiring exact attention (e.g., precise 
retrieval), this is a limitation.

## 10.2 Task Coverage

Only SST-2 is evaluated. Long-context tasks and generation tasks are not 
tested.

## 10.3 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.4 Hardware

All experiments on RTX 4070 Laptop. Not tested on:
- A100/H100
- CPU
- Edge devices

## 10.5 No Activation Weighting

Unlike A3 and AASVD, SVD-Linear uses unweighted SVD. Adding activation-aware 
weighting could improve results.

## 10.6 Streaming Overhead

`step_recurrent` has Python-level loop overhead. For very short sequences, 
batch mode is faster.

## 10.7 Cross-Attention Not Supported

`forward` accepts `encoder_hidden_states` but does not use it. Cross-attention 
is effectively unsupported.

## 10.8 Chunk Size Fixed at 64

The chunk size is not adaptive. Larger chunks give more parallelism but 
reduce the effect of decay.

## 10.9 Decay Initialization Approximation

The bias initialization `b = -log(decay_times)` is an approximation. The 
actual decay is $\lambda = e^{-\text{softplus}(b)}$, which is not exactly 
the target. We do not correct for this.

## 10.10 Single-Seed Results

All main results from a single seed.

## 10.11 No Theoretical Bound on Downstream Accuracy

We provide no bound on the approximation error vs softmax.

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade implementation. Known issues:
- Python loop in chunked scan
- `raise` in `copy_weights`
- No unit tests for numerical edge cases
- Padding handling has multiple branches

---