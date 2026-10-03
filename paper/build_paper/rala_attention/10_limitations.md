<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compression Method

RALA **increases** parameter count by **48.5\%%**. It is an 
attention replacement, not a compressor.

## 10.2 Sequential Prefix Scan

The causal prefix scan uses a Python loop over sequence length. This 
dominates runtime and is unsuitable for production. A fused parallel scan 
is needed.

## 10.3 Frozen $U, V$

Our implementation freezes the augmentation matrices $U, V$ when LoRA is 
active. This means the rank augmentation is a **fixed random projection 
scaled by $\alpha$**, not a learned matrix. This is a significant 
limitation.

## 10.4 Scalar Gate

A single scalar $\alpha$ per layer is shared across all heads and all 
positions. This is minimal but may not be optimal.

## 10.5 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.6 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed. No >1B models.

## 10.7 No Fused Kernel

We do not provide a Triton or CUDA implementation of the prefix scan.

## 10.8 Single Seed

Single seed. No confidence intervals.

## 10.9 No Comparison with State-of-the-Art Linear Attention

We do not compare against Delta Net, GLA, Mamba, or RWKV. Our comparison is 
limited to standard linear attention with elu+1.

## 10.10 Numerical Stability

Prefix sums in FP16 may lose precision for $L > 512$. Consider FP32 for long 
sequences.

## 10.11 Code Quality

Research-grade:
- Broad exception handling in `inject_rala`
- No unit tests for kernel features
- Some warnings instead of errors for weight copy failure

## 10.12 What This Paper Is Not

This is **not**:
- A compression method
- A production-ready attention
- A state-of-the-art result
- A scaling study

---