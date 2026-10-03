<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

RPC **increases** parameter count by ~48.5\%% due to LoRA. It 
does not reduce storage, memory, or model size.

## 10.2 No Memory Reduction

RPC **does not reduce memory**. The score matrix $S^{RPC}$ remains $L \times L$. 
FlashAttention-style memory savings require not materializing $S$, which 
RPC does not change.

## 10.3 Only Helps at Long Sequences

RPC is slower than baseline at $L \leq 256$ due to overhead. Benefit appears 
at $L > 512$.

## 10.4 Approximation Error

The JL-based sketch is lossy. RPC scores are only approximately equal to 
true scores. For tasks requiring exact attention (e.g., retrieval), this 
is a limitation.

## 10.5 Task Coverage

Only SST-2 is evaluated.

## 10.6 Model Coverage

Tested on ≤124M parameter models. Not tested on >1B models.

## 10.7 Hardware

All experiments on RTX 4070 Laptop.

## 10.8 Cross-Attention Not Supported

RPC's forward accepts `encoder_hidden_states` but does not use it.

## 10.9 Nested Encoder Complexity

RoBERTa requires `is_nested_encoder = True`.

## 10.10 `raise` on Failure

RPC raises `RuntimeError` on any injection failure. This is safer than 
silent fallback but less flexible.

## 10.11 Single-Seed Results

All main results from a single seed.

## 10.12 No Theoretical Bound on Downstream Accuracy

We provide a JL-based bound on score error (Appendix A) but no bound on 
downstream task error.

## 10.13 Unresolved Failures

0.85

## 10.14 Code Quality

Research-grade implementation. Known issues:
- `raise` in `copy_weights` may be too aggressive
- No unit tests for edge cases

---