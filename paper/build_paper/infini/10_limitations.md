<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Compressor

Infini-Attention **increases** parameter count by ~48.5\%%. It is 
designed for **inference efficiency at long sequences**, not for model 
size reduction.

## 10.2 Task Coverage

Only SST-2 is evaluated. Infini-Attention may behave differently on:

- Generation tasks (summarization, translation)
- Long-context reasoning (needle-in-haystack)
- Multi-class classification

## 10.3 Model Coverage

Tested on models with ≤124M parameters. Not tested on:

- >1B parameters
- Very deep models (>24 layers)
- Mixture-of-Experts

## 10.4 Hardware

All experiments on RTX 4070 Laptop. Not tested on:

- A100/H100
- CPU
- Edge devices

## 10.5 Cross-Attention Not Supported

Infini-Attention raises `NotImplementedError` for cross-attention. This 
excludes encoder-decoder models (T5, BART).

## 10.6 Nested Encoder Complexity

RoBERTa requires correct `is_nested_encoder = True`. If mis-set, 
double-projection causes scale collapse.

## 10.7 Inference Overhead

At $L = 128$, Infini-Attention is slower than baseline due to the extra 
matmuls. The benefit materializes only at $L \gg 256$.

## 10.8 Memory Interpretation

We do not analyze what the memory learns. In particular:

- Does it capture global sentiment?
- Does it memorize specific tokens?
- How does it interact with position embeddings?

## 10.9 Single-Seed Results

All main results from a single seed. SE ~1.69%. Differences below 3.4% 
may be noise.

## 10.10 No Theoretical Guarantee

We provide no bound on approximation error vs the baseline attention. 
The memory is a learned parametric approximator.

## 10.11 No Comparison with Compressive Transformer

We do not run head-to-head comparison with Compressive Transformer (Rae et 
al., 2020) or Recurrent Memory Transformer (Bulatov et al., 2022).

## 10.12 Unresolved Failures

0.85

## 10.13 Code Quality

Research-grade implementation. Known issues:

- No unit tests
- Broad exception handling
- NaN not explicitly handled in memory branch

---