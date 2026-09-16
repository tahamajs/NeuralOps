# 1. Introduction

## 1.1 Motivation

Linear attention (Katharopoulos et al., 2020) computes attention in 
$\mathcal{O}(L)$ time by replacing softmax with a positive kernel feature 
map $\phi$:

$$
O_t = \frac{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j) v_j^\top}{\phi(q_t)^\top \sum_{j \leq t} \phi(k_j)}
\tag{1.1}
$$

The state $S_t = \sum_{j \leq t} \phi(k_j) v_j^\top \in \mathbb{R}^{F \times d_h}$ 
is updated incrementally, giving constant memory per step.

**Two practical problems remain:**

1. **Uniform contribution of past keys.** In (1.1), the coefficient of 
   $v_j$ in $O_t$ is $\phi(q_t)^\top \phi(k_j) / Z_t$ — it does **not** 
   decay with $t - j$. Long-range dependencies and recent context are 
   treated identically. In practice, transformer attention has been shown 
   to concentrate on nearby tokens (attention sinks aside).
2. **Causal implementations often use cumsum.** Naive causal linear attention 
   materializes $S_t$ for every $t$, giving an $\mathcal{O}(L \cdot F \cdot d_h)$ 
   intermediate. This defeats the memory advantage.

**DecayLinearAttention** addresses both:

1. **Data-dependent decay.** We learn a per-head, per-token decay 
   $\lambda_{h,t} \in (0, 1)$ and update the state with exponential decay:
   
   $$
   S_t = \lambda_{h,t} S_{t-1} + \phi(k_t) v_t^\top
   \tag{1.2}
   $$
   
2. **Chunked scan with $\mathcal{O}(1)$ state.** We carry only 
   $S \in \mathbb{R}^{d_h \times d_h}$ between chunks, processing each 
   chunk in parallel.

## 1.2 Problem Statement

Given a pretrained transformer with softmax attention $\mathcal{A}$, we seek 
a linear attention $\mathcal{A}'$ such that:

1. **Linear complexity:** $\mathcal{A}'$ runs in $\mathcal{O}(L)$ time.
2. **O(1) memory during causal decoding.** No $\mathcal{O}(L)$ intermediate.
3. **Data-dependent decay.** The decay rate depends on the input.
4. **Bidirectional & causal support.**
5. **Cross-attention support.**
6. **Pretrained compatibility** via LoRA recovery.

## 1.3 Our Approach

Three components:

### 1.3.1 Data-Dependent Exponential Decay

$$
\lambda_{h,t} = \sigma\!\left(W_g x_t + b_g + \ell_h\right), \quad
\ell_h \in [{{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}]
\tag{1.3}
$$

The per-head bias $\ell_h$ is initialized to linearly spaced values, so 
different heads start with different context horizons.

### 1.3.2 Chunked Scan

For a chunk of length $C$, we compute:

- **Intra-chunk:** Dense causal attention within the chunk.
- **Inter-chunk:** State-passing from previous chunks via $S$.

Total memory: $\mathcal{O}(C^2 + d_h^2)$ per chunk, **independent of $L$**.

### 1.3.3 ELU+1 Feature Map

$$
\phi(x) = \text{elu}(x) + 1
\tag{1.4}
$$

This is positive (needed for stable linear attention) and cheap to compute.

## 1.4 Contributions

- **Data-dependent decay.** A per-head, per-token decay rate learned from 
  the input.
- **Chunked scan with O(1) state.** Memory independent of sequence length.
- **Bidirectional and causal support.** Symmetric average of forward and 
  backward scans for encoders.
- **Cross-attention support.** All four projections factorized.
- **Multi-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen3.5.
- **Honest memory reporting.** We document the exact memory footprint.
- **Full reproducibility.** Code, weights, and scripts released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on linear 
attention, exponential decay, and chunked scans. Section 4 presents 
DecayLinearAttention. Section 5 describes integration. Section 6 details 
the setup. Section 7 reports results. Section 8 analyzes decay learning. 
Section 9 discusses trade-offs. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $C$ | Chunk size (32) |
| $\lambda_{h,t}$ | Decay rate at position $t$, head $h$ |
| $\ell_h$ | Per-head decay bias |
| $S$ | Inter-chunk state |
| $z$ | Inter-chunk denominator state |
| $\phi$ | Feature map |
| $\epsilon$ | Numerical stability constant |
```

---