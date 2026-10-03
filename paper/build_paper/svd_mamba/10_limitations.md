<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

SVD-Mamba **increases** parameter count by ~48.5\%%. While SVD 
factorization reduces projection parameters, the SSM branch, global 
mixing, gating, and LoRA add more.

## 10.2 Complex Architecture

SVD-Mamba has three paths (attention, SSM, global) fused by a gate. This 
is much more complex than standard attention, making debugging and 
deployment harder.

## 10.3 Unstable Recovery

The gate bias $-7$ keeps the SSM silent initially, but recovery training 
may shift it. This creates training instability.

## 10.4 Numerical Sensitivity

The SSM scan requires multiple clamps to prevent overflow. Without careful 
numerical handling, the branch produces NaN.

## 10.5 Inference Overhead

At short sequences ($L \leq 256$), SVD-Mamba is slower than baseline.

## 10.6 Cross-Attention Not Supported

If `encoder_hidden_states` is provided, SVD-Mamba returns the input 
unchanged — effectively a no-op. Encoder-decoder models (T5, BART) are 
not supported.

## 10.7 Task Coverage

Only SST-2 is evaluated. Long-context tasks (where SSM should help) are 
not tested.

## 10.8 Model Coverage

Tested on ≤124M parameter models.

## 10.9 Hardware

All experiments on RTX 4070 Laptop.

## 10.10 Silent Injection Failure

The injection hook catches exceptions and returns `None`. If partially 
successful, this can lead to broken models without explicit errors.

## 10.11 Single-Seed Results

All main results from a single seed.

## 10.12 No Theoretical Bound

No bound on approximation error vs the baseline attention.

## 10.13 Unresolved Failures

0.85

## 10.14 Code Quality

Research-grade implementation. Known issues:

- Complex forward with many clamps
- Python loop over chunks
- `raise` in `copy_weights` but `return None` in `inject`
- No unit tests

---