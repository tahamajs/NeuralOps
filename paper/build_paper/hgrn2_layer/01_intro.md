# 1. Introduction

## 1.1 Motivation

The quadratic complexity of transformer attention, $\mathcal{O}(L^2)$ in 
sequence length, is the primary bottleneck for long-context inference. Two 
families of solutions have emerged:

1. **Sparse attention** reduces the number of attended pairs but retains 
   $L \times k$ memory patterns.
2. **Linear recurrence** replaces attention entirely with a fixed-size state 
   that is updated sequentially.

Within the second family, **HGRN** (Hierarchically Gated Recurrent Network) 
and its successor **HGRN-2** (Qin et al., 2024) stand out for their 
**element-wise** recurrence: each feature dimension maintains its own scalar 
state, updated by an input-dependent forget gate. This is simpler than 
matrix-state alternatives (DeltaNet, Mamba) and runs in $\mathcal{O}(L d)$ 
time.

HGRN2Layer is our implementation of the HGRN-2 recurrence, packaged as a 
drop-in attention replacement for existing pretrained transformers.

## 1.2 Problem Statement

Given a pretrained transformer with attention module $\mathcal{A}$, we seek 
a replacement $\mathcal{A}'$ such that:

1. **Linear complexity:** $\mathcal{A}'$ runs in $\mathcal{O}(L d)$ time and 
   $\mathcal{O}(d)$ memory.
2. **Bidirectional or causal:** Supports both encoder (bidirectional) and 
   decoder (causal) modes.
3. **Pretrained compatibility:** Base weights transfer from $\mathcal{A}$.
4. **Learnable recovery:** Only LoRA adapters and gate networks are trained.

We do **not** require parameter reduction. The goal is **linear-time 
inference**.

## 1.3 Our Approach

HGRN2Layer maintains a per-feature scalar state $s_t \in \mathbb{R}^d$ 
updated by:

$$
s_t = f_t \odot s_{t-1} + (1 - f_t) \odot i_t \odot v_t
\tag{1.1}
$$

where:

- $f_t = \exp(-\text{softplus}(W_f x_t)) \in [f_\text{min}, 1]$ is the 
  **bounded forget gate**.
- $i_t = \sigma(W_i x_t)$ is the **input gate**.
- $v_t = W_v x_t$ is the **value** projection.

The critical design choices are:

1. **Bounded forget gate:** $f_t \in [10^{-3}, 1]$ prevents exponential 
   underflow/overflow.
2. **Input modulation:** Multiplying $v_t$ by $(1 - f_t)$ ensures that when 
   the state forgets, it also receives a proportional update — this avoids 
   sign-flipping.
3. **Parallel chunked scan:** The recurrence is computed by chunking the 
   sequence (chunk size 32) and running a cumulative-sum-based associative 
   scan in FP32.
4. **Bidirectional support:** For encoders, we run both forward and backward 
   scans and sum the results.

## 1.4 Contributions

- **A production-hardened HGRN-2 implementation.** We implement the 
  element-wise recurrence with bounded forget gates, parallel chunked scan, 
  and bidirectional/causal support.
- **Robust weight transfer.** Our `copy_weights` handles PyTorch's native 
  `MultiheadAttention`, GPT-2 Conv1D, RoBERTa nested, Qwen GQA, and 
  DistilBERT flat attention.
- **An honest evaluation.** We report both the long-context speedup and the 
  short-context slowdown of the current implementation.
- **Full reproducibility.** Code, weights, and evaluation scripts are 
  released under Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work in linear recurrence and SSMs. Section 3 
provides background on HGRN-2, gated recurrences, and parallel scan. 
Section 4 presents HGRN2Layer. Section 5 describes integration with four 
architectures. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the trade-offs. Section 9 discusses guidance. Section 10 
lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $s_t$ | Element-wise state |
| $f_t$ | Forget gate |
| $i_t$ | Input gate |
| $v_t$ | Value projection |
| $z_t$ | Output gate input |
| $\sigma$ | Sigmoid function |
| $\odot$ | Element-wise product |

---