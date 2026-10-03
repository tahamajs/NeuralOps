# 1. Introduction

## 1.1 Motivation

Hybrid attention-SSM architectures have emerged as a promising direction for 
efficient transformers (Lieber et al., 2024; Dong et al., 2024). They 
combine:

- **Attention** for precise, content-based retrieval.
- **State space models** for linear-time long-range mixing.

A common design choice in these architectures is to share the same input 
between the two branches: the SSM consumes the full hidden state 
$x \in \mathbb{R}^d$, just like the attention branch.

**But this is information-theoretically wasteful.** The SSM's recurrent 
state has fixed dimension, and its ability to encode information is bounded. 
Feeding it a $d$-dimensional input with high redundancy does not increase 
its effective capacity — it just adds noise and parameters.

## 1.2 The Information Bottleneck Perspective

The Information Bottleneck (IB) principle (Tishby et al., 1999) formalizes 
the idea that a good representation $Z$ of an input $X$ for a task $Y$ 
should:

1. **Preserve** information about $Y$: $\max I(Z; Y)$
2. **Compress** information about $X$: $\min I(Z; X)$

For an SSM branch, the "task" is long-range context modeling. A bottleneck 
$Z = f(X)$ with $|Z| < |X|$ forces the SSM to encode only the essential 
concepts, discarding redundancy.

## 1.3 Problem Statement

Given an input $X \in \mathbb{R}^{B \times L \times d}$ and a hybrid block 
with an SSM branch, we ask: **should the SSM see the full $X$ or a 
compressed version?**

We hypothesize that the compressed version is better, and formalize this as:

$$
\text{SSM}_\text{concept}(X) = \text{SSM}\!\left(W_\text{up} \cdot \text{GELU}(W_\text{down} X)\right)
\tag{1.1}
$$

with $W_\text{down} \in \mathbb{R}^{b \times d}$, $b \ll d$.

## 1.4 Our Approach: JambaInfo

JambaInfo combines three ideas:

### 1.4.1 Differential Attention Branch

Two Q-K pairs, ReLU-damped subtraction (Ye et al., 2024):

$$
A_\text{diff} = \text{ReLU}(A_1 - 0.2 A_2)
\tag{1.2}
$$

### 1.4.2 Concept-Bottlenecked SSM Branch

The SSM input is compressed through a low-rank concept bottleneck:

$$
z = W_\text{up} \cdot \text{GELU}(W_\text{down} x)
\tag{1.3}
$$

with $b = 256$.

### 1.4.3 Learned Fusion

$$
\text{out} = (1 - \alpha) \odot A_\text{diff} + \alpha \odot M_\text{concept}
\tag{1.4}
$$

with $\alpha = \sigma(w)$ and $w_0 = -2$ (favoring attention initially).

## 1.5 Contributions

- **An Information-Bottlenecked SSM branch.** We introduce a concept 
  bottleneck between the hidden state and the SSM, motivated by IB theory.
- **Dual operational modes.** JambaInfo supports a "retrieval" mode (default) 
  and a "compression" mode, configurable per layer.
- **A production-hardened implementation.** We handle four architectures, 
  with strict weight-transfer validation.
- **An honest evaluation.** We report the accuracy, parameter cost, and 
  bottleneck-dimension sensitivity.
- **Full reproducibility.** Code, weights, and scripts released.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on IB, 
attention, and SSMs. Section 4 presents JambaInfo. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the bottleneck. Section 9 discusses guidance. Section 10 
lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $b$ | Bottleneck dimension |
| $H$ | Number of heads |
| $d_h$ | Head dim |
| $A_i$ | Attention map |
| $\lambda$ | Diff damping ($= 0.2$) |
| $\alpha$ | Fusion gate |
| $z$ | Concept representation |

---