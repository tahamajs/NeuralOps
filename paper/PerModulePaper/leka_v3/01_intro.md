# 1. Introduction

## 1.1 Motivation

Transformer self-attention scales quadratically with sequence length. For 
input $X \in \mathbb{R}^{B \times L \times d}$ with $H$ heads and head 
dimension $d_h$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

The intermediate matrix $QK^\top \in \mathbb{R}^{B \times H \times L \times L}$ 
costs $\mathcal{O}(L^2)$ time and memory. At $L = 4096$ with $H = 12$ heads 
and batch size 1, this is ~200M entries — over 800 MB at FP16.

**Linear attention** addresses this by factorizing the softmax kernel:

$$
e^{q \cdot k} \approx \phi(q)^\top \phi(k)
\tag{1.2}
$$

which allows computing $K^\top V$ first (cost $\mathcal{O}(L d_h d_h)$) then 
multiplying by $Q$ (cost $\mathcal{O}(L d_h d_h)$). This achieves $\mathcal{O}(L)$ 
complexity.

However, naive linear attention suffers from two issues:

1. **Approximation error:** Random Fourier features approximate softmax poorly 
   for peaked distributions.
2. **Loss of locality:** Linear attention has a global receptive field, missing 
   the strong local structure in language.

**LEKAv3** combines the two approaches:

- A **global branch** computes linear-attention-like kernel attention over 
  a small set of landmarks, capturing global context.
- A **local branch** computes exact attention within a sliding window, 
  preserving local detail.
- A **learnable gate** fuses the two.

## 1.2 Problem Statement

Given a sequence of length $L$, we seek an attention mechanism $\mathcal{A}'$ 
with:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\text{Cost}(\mathcal{A}') = \mathcal{O}(L \cdot c)
\tag{1.3}
$$

where $c$ is a constant (independent of $L$). We additionally require:

1. **Bounded approximation error:** Softmax output approximates the exact one.
2. **Causality:** For decoders, query at position $i$ attends only to keys 
   at positions $\leq i$.
3. **GQA compatibility:** Handles $H_{kv} < H$.
4. **Multi-architecture:** Works for flat, nested, and causal models.

## 1.3 Our Approach

LEKAv3 has three components:

### 1.3.1 Multi-Scale Kernel Features

We map each $Q, K$ vector to a feature space via cos/sin projections:

$$
\phi(x) = \frac{1}{\sqrt{d_\text{feat}}} \begin{pmatrix} \cos(W_1 x) \\ \sin(W_1 x) \end{pmatrix}
\tag{1.4}
$$

where $W_1 \in \mathbb{R}^{d_h \times d_\text{feat}/2}$ is a random projection.

### 1.3.2 Value-Scored Landmarks

Instead of using all $L$ keys, we select $L_\text{land}$ landmarks using a 
learnable scorer:

$$
s_i = \text{MLP}(K_\text{feat}[i])
\tag{1.5}
$$

The top-$L_\text{land}$ indices are selected. Token 0 is always retained 
(attention sink).

### 1.3.3 Gated Local + Global Fusion

The final output is:

$$
\text{Out} = \sigma(g) \cdot \text{Out}_\text{land} + (1 - \sigma(g)) \cdot \text{Out}_\text{local}
\tag{1.6}
$$

where $g$ is a per-head learnable gate.

## 1.4 Contributions

- **A landmark-based linear attention.** LEKAv3 approximates softmax attention 
  at $\mathcal{O}(L \cdot L_\text{land} + L \cdot W)$ complexity.
- **A learnable landmark scorer.** The scorer is a small MLP trained jointly 
  with LoRA recovery, learning content-dependent landmark selection.
- **Multi-scale kernel features.** We use two projection scales to better 
  capture both fine and coarse structure.
- **A production-hardened implementation.** Handles DistilBERT, RoBERTa, 
  GPT-2, and Qwen with GQA, RoPE, nested encoders, and causal masking.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on kernel 
attention, landmarks, and RFF. Section 4 presents LEKAv3. Section 5 
describes integration. Section 6 details the experimental setup. Section 7 
reports results. Section 8 analyzes the contribution of each component. 
Section 9 discusses trade-offs. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $L_\text{land}$ | Number of landmarks |
| $W$ | Local window size |
| $H, H_{kv}$ | Query, KV heads |
| $d_h$ | Head dimension |
| $d_\text{feat}$ | Kernel feature dimension |
| $\phi$ | Kernel feature map |
| $s_i$ | Landmark importance score |
| $\mathcal{L}$ | Landmark index set |
| $\sigma(g)$ | Gate value |

---