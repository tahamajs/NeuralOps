<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Dense Softmax Attention

For $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{L \times d_h}
\tag{3.1}
$$

**Complexity.** $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory.

## 3.2 Token Pruning

Restrict attention to a subset $\mathcal{T} \subseteq \{1, \dots, L\}$ 
of **retained tokens**:

$$
\text{Attn}_\text{pruned}(Q, K, V) = \text{softmax}\!\left(\frac{Q_\mathcal{T} K_\mathcal{T}^\top}{\sqrt{d_h}}\right) V_\mathcal{T}
\tag{3.2}
$$

where subscript $\mathcal{T}$ denotes indexing.

**Complexity.** $\mathcal{O}(L |\mathcal{T}| d_h)$ time, $\mathcal{O}(L |\mathcal{T}|)$ memory.

For $|\mathcal{T}| = \rho L$:

$$
\text{Complexity reduction} = \rho
\tag{3.3}
$$

## 3.3 Scoring Network

A learnable function $f_\theta: \mathbb{R}^d \to \mathbb{R}$ maps hidden 
states to scores. Choices:

| Type | Expression | Params |
|---|---|---|
| Linear | $w^\top x$ | $d$ |
| MLP-1 | $W_2 \cdot \text{GELU}(W_1 x)$ | $64 d$ |
| MLP-2 | Two hidden layers | More |

PTP uses **MLP-1** (one hidden layer of 64 units). This is a good 
trade-off between expressiveness and parameter count.

## 3.4 Top-K Selection

Given scores $s \in \mathbb{R}^L$:

$$
\mathcal{T} = \text{top-}k(s)
\tag{3.4}
$$

**Complexity.** $\mathcal{O}(L \log k)$ via heap or $\mathcal{O}(L)$ via 
quickselect.

## 3.5 [CLS] Preservation

In classification tasks, the [CLS] token (position 0) carries the 
aggregated sequence representation. Dropping it breaks classification.

**PTP guarantees [CLS] retention** by setting $s_0 = 10^4$ before top-$k$:

$$
s_0 \leftarrow 10^4
\tag{3.5}
$$

This is a soft guarantee (no runtime branching required).

## 3.6 Padding Handling

Padding tokens have zero or invalid content. They must be excluded:

$$
s_i \leftarrow -10^4 \quad \text{if token } i \text{ is padding}
\tag{3.6}
$$

## 3.7 Causal Masking

For decoder models, position $i$ cannot attend to $j > i$. After pruning, 
we must preserve this by comparing **original** positions:

$$
\text{mask}_{ij} = \mathbb{1}[\text{orig}(j) > i]
\tag{3.7}
$$

## 3.8 Rotary Position Embeddings

RoPE (Su et al., 2021) applies position-dependent rotation:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \text{rot}(Q_i) \sin(\theta_i)
\tag{3.8}
$$

PTP applies RoPE to Q, K before pruning.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $\rho$ | Keep ratio |
| $k$ | Retained tokens |
| $s_i$ | Score of token $i$ |
| $\mathcal{T}$ | Retained indices |
| $f_\theta$ | Score network |

---