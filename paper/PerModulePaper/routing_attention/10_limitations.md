<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not Faster Than Dense Attention

The naive loop over clusters computes $K$ full attention matrices. This is 
**slower** than dense attention by a factor of $K$. Real speedups require 
sparse matmul kernels.

## 10.2 Non-Differentiable Cluster Assignment

k-means uses `argmin`, which is not differentiable. Gradients do not flow 
through the cluster assignment.

## 10.3 Not a Compression Method

LoRA adapters add **48.5\%%** parameters. Routing is a computational 
substitution, not a compressor.

## 10.4 K-Means May Degrade on Padded Inputs

Padding tokens are included in clustering. This may distort centroids.

## 10.5 Mask-K Independence

All queries in cluster $k$ attend to the same set of keys. This limits 
expressiveness relative to query-specific sparse patterns.

## 10.6 Token-0 Anchor is a Hack

The token-0 anchor is a fixed prior. Its effectiveness depends on the 
attention-sink phenomenon, which may not hold for all models or tasks.

## 10.7 RuntimeError on Zero Norm

Our `copy_weights` raises `RuntimeError` if any projection has near-zero 
norm. This can abort injection for the entire model. Should be a warning.

## 10.8 No Double-Injection Guard

Running `inject_routing` twice may corrupt the model.

## 10.9 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.10 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed.

## 10.11 Single Seed

No multi-seed runs. Differences below 3.4% may be noise.

## 10.12 No Sparse Kernel

We do not provide a Triton or CUDA implementation of sparse attention.

## 10.13 Comparison with Published Sparse Methods

We do not compare head-to-head with Reformer, Routing Transformer, or 
BigBird. Our analysis is limited to internal comparisons.

## 10.14 Numerical Stability

FP16 attention scores may overflow for long sequences. Consider FP32 for 
$L > 512$.

## 10.15 What This Paper Is Not

This is **not**:
- A production sparse attention
- A compression method
- An efficiency study
- A state-of-the-art result

---