<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Multi-Head Self-Attention

For $X \in \mathbb{R}^{B \times L \times d}$:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{3.1}
$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$, and:

$$
\text{MHSA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.2}
$$

**Complexity:** $\mathcal{O}(L^2 d)$.

## 3.2 Sliding-Window Attention

Restrict attention to a window of size $W$:

$$
\text{SWA}(X)_i = \text{softmax}\!\left(\frac{Q_i K_{[i-W:i+W]}^\top}{\sqrt{d_h}}\right) V_{[i-W:i+W]}
\tag{3.3}
$$

For causal models, the window is one-sided. Complexity: $\mathcal{O}(L W d)$.

## 3.3 Soft-Attention Pooling

Given a sequence $X \in \mathbb{R}^{B \times L \times d}$ and a learnable 
projection $W_p \in \mathbb{R}^{d \times d_c}$, the soft-attention pooled 
representation is:

$$
\text{Pool}_j = \sum_{i=1}^{L} \alpha_{ij} X_i, \qquad
\alpha_{ij} = \frac{\exp(W_{p,j}^\top X_i)}{\sum_{i'=1}^L \exp(W_{p,j}^\top X_{i'})}
\tag{3.4}
$$

for $j = 1, \dots, d_c$.

**Interpretation.** $W_{p,j}$ is a "query" for the $j$-th summary slot. 
$\alpha_{ij}$ is the weight of token $i$ in slot $j$. The pooled 
representation $\text{Pool}_j$ is a weighted average of tokens.

## 3.4 Gated Fusion

Given two branches (local and global), we combine them via a gate:

$$
\text{output} = \gamma \odot \text{local} + (1 - \gamma) \odot \text{global}
\tag{3.5}
$$

with $\gamma = \sigma(W_g X) \in [0, 1]$.

**Initialization.** With bias $= 2.0$, $\sigma(2) \approx 0.88$ at start, 
favoring the local branch.

## 3.5 Grouped-Query Attention (GQA)

For GQA with $H_{kv} < H$, we repeat $K, V$ before attention.

## 3.6 Rotary Position Embeddings (RoPE)

RoPE rotates Q, K by position-dependent angles.

## 3.7 LoRA

LoRA freezes $W$ and adds $\frac{\alpha}{r} B A$ with $B = 0$ at 
initialization.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $W$ | Window size |
| $d_c$ | Compression rank |
| $H$ | Query head count |
| $H_{kv}$ | KV head count |
| $d_h$ | Per-head dimension |
| $C_K, C_V$ | Compressed summary keys, values |
| $\gamma$ | Gate |
| $r$ | LoRA rank |

---
