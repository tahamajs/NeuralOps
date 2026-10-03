# 1. Introduction

## 1.1 Motivation

Attention and recurrence have complementary failure modes:

1. **Attention** captures precise pairwise dependencies via softmax scores, 
   but scales $\mathcal{O}(L^2)$ and often becomes **noisy** — a fraction 
   of tokens receive most attention mass, while others contribute spurious 
   signal.
2. **State space models** capture long-range context in $\mathcal{O}(L)$ 
   with a fixed-size state, but lack the **content-based retrieval** of 
   attention.

Recent work has shown that both can be improved:

- **Differential Transformer** (Ye et al., 2024) uses two Q-K projections and 
  subtracts the two attention maps, cancelling common-mode noise.
- **Hymba** (Dong et al., 2024) fuses attention and SSM inside a single block 
  with a learned gate.

We combine both ideas in **HymbaTransX**.

## 1.2 Problem Statement

Given an input $X \in \mathbb{R}^{B \times L \times d}$, we seek a hybrid 
attention block $\mathcal{B}(X)$ that:

1. **Reduces attention noise** relative to standard softmax attention.
2. **Provides long-range context** via a recurrent state.
3. **Scales linearly** in sequence length for the SSM branch.
4. **Supports GQA, RoPE, and causal masking**.
5. **Transfers pretrained weights** from a standard attention block.

We do **not** require parameter reduction. The goal is **quality improvement 
at roughly constant parameter count**.

## 1.3 Our Approach

HymbaTransX has three components:

### 1.3.1 Differential Attention

Two query-key pairs:

$$
Q_1 = X W_{Q_1}, \quad K_1 = X W_{K_1}, \quad
Q_2 = X W_{Q_2}, \quad K_2 = X W_{K_2}
\tag{1.1}
$$

Two attention patterns:

$$
A_1 = \text{softmax}\!\left(\frac{Q_1 K_1^\top}{\sqrt{d_h}}\right), \quad
A_2 = \text{softmax}\!\left(\frac{Q_2 K_2^\top}{\sqrt{d_h}}\right)
\tag{1.2}
$$

Differential output:

$$
A_\text{diff} = \text{ReLU}(A_1 - \lambda A_2)
\tag{1.3}
$$

with $\lambda = 0.2$. The subtraction cancels the shared component of the 
two attention patterns, keeping only the "signal".

### 1.3.2 State Space Model Branch

A `SimpleSSM` branch produces $M_\text{ssm} \in \mathbb{R}^{B \times L \times d}$ 
via linear recurrence. For decoders, a forward pass. For encoders, both 
forward and backward, averaged.

### 1.3.3 Learned Fusion

A single scalar gate $\alpha = \sigma(w)$ blends the two branches:

$$
\text{out} = \alpha \odot A_\text{diff} + (1 - \alpha) \odot M_\text{ssm}
\tag{1.4}
$$

Initialized to $\alpha = \sigma(-1) \approx 0.27$ (favoring SSM initially).

## 1.4 Contributions

- **A hybrid differential + SSM block.** We combine two emerging ideas — 
  differential attention and attention-SSM fusion — in a single module.
- **Internal RoPE fallback.** When the parent model does not provide 
  position embeddings, we compute them via `HeadRoPE`.
- **A production-hardened implementation.** We handle DistilBERT, RoBERTa, 
  GPT-2, and Qwen3.5, with strict weight-transfer validation and safe 
  fallback.
- **An honest evaluation.** We report both the accuracy benefit and the 
  parameter cost.
- **Full reproducibility.** Code, weights, and evaluation scripts released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
differential attention, and SSMs. Section 4 presents HymbaTransX. Section 5 
describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes the contributions of each component. Section 9 
discusses guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $H$ | Number of heads |
| $H_{kv}$ | KV heads |
| $d_h$ | Head dimension |
| $Q_i, K_i$ | Query and key for branch $i$ |
| $A_i$ | Attention pattern for branch $i$ |
| $\lambda$ | Differential damping ($= 0.2$) |
| $\alpha$ | Fusion gate |

---