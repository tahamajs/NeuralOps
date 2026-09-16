# 1. Introduction

## 1.1 Motivation

Softmax attention has been the dominant attention mechanism since 
Vaswani et al. (2017). It is well-behaved, but its **non-kernelizability** 
forces the computation of the full $[N \times N]$ attention matrix. This 
is why standard attention is quadratic.

**Polynomial alternatives.** A longstanding observation is that the 
**softmax kernel** $e^{q \cdot k}$ can be approximated by a **polynomial**:

$$
e^x \approx \sum_{n=0}^{d} \frac{x^n}{n!}
\tag{1.1}
$$

Truncating this at order $d$ yields a polynomial kernel that **is** 
kernelizable:

$$
(1 + x)^d = \sum_{n=0}^{d} \binom{d}{n} x^n
\tag{1.2}
$$

This is the basis of **polynomial attention** (Choromanski et al., 2021; 
Kacham et al., 2024). If successful, it enables $\mathcal{O}(N)$ attention 
via random-feature or explicit polynomial-feature maps.

**PADRe** explores this polynomial substitution in the context of **pretrained** 
transformers. We ask: can we replace softmax attention with a polynomial 
kernel without retraining from scratch?

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with softmax attention, we 
seek a replacement attention $\mathcal{A}'$ such that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\mathcal{A}' \text{ uses a polynomial kernel}
\tag{1.3}
$$

subject to:

1. **Weight reuse:** Pretrained Q, K, V, O weights must be preserved.
2. **Numerical stability:** No NaN/overflow during training.
3. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2, Qwen.
4. **Causal correctness:** Preserve causality for decoders.
5. **Recovery compatibility:** LoRA fine-tuning on top.

## 1.3 Our Approach

PADRe replaces softmax with:

$$
\text{attn}(q_i, k_j) = \frac{(1 + s_{ij}/\text{order})^\text{order}}{\sum_{j'} (1 + s_{ij'}/\text{order})^\text{order}}
\tag{1.4}
$$

where $s_{ij} = (q_i \cdot k_j)/\sqrt{d_h} - m_i$ and $m_i = \max_j s_{ij}$ over valid $j$.

**Numerical stability tricks:**

1. **Max-centering** of scores over valid positions:
   $$
   s_{ij} \leftarrow s_{ij} - \max_j s_{ij}
   \tag{1.5}
   $$
   This ensures $s_{ij} \leq 0$, so the base $1 + s_{ij}/\text{order}$ is 
   bounded from above by 1.

2. **ReLU-clamping** of the base:
   $$
   \text{base} = \text{ReLU}(1 + s_{ij}/\text{order})
   \tag{1.6}
   $$
   Prevents negative base raised to a non-integer power.

3. **Mask handling:** Invalid positions (padding, causal future) are zeroed 
   after exponentiation.

## 1.4 Contributions

- **A polynomial-kernel attention module** that replaces softmax and can be 
  applied to pretrained transformers.
- **Numerical stability via max-centering**, preventing overflow in the 
  polynomial expansion.
- **Multi-architecture weight transfer** with Frobenius validation.
- **Output-equivalence check** during injection, skipping layers that fail.
- **Honest complexity analysis.** We acknowledge that without kernelization, 
  runtime is still $\mathcal{O}(N^2 d_h)$.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on softmax 
and polynomial approximations. Section 4 presents PADRe. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. Section 
8 analyzes trade-offs. Section 9 discusses usage. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $s_{ij}$ | Centered scaled score |
| $m_i$ | Max score over valid positions |
| $d$ | Polynomial order |
| $\phi$ | Polynomial feature map |

---