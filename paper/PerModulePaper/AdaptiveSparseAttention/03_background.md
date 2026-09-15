<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

For a single head with $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
S = \frac{Q K^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times L}
\tag{3.1}
$$

$$
P = \text{softmax}(S) \in \mathbb{R}^{L \times L}
\tag{3.2}
$$

$$
O = P V \in \mathbb{R}^{L \times d_h}
\tag{3.3}
$$

**Complexity.** $S$ and $P$ require $\mathcal{O}(L^2)$ memory. The two 
matmuls require $\mathcal{O}(L^2 d_h)$ time. For $L = 1024$, $d_h = 64$, 
this is $\sim 67$M FLOPs per head per sequence.

## 3.2 Sparsity in Attention

**Observation.** The softmax distribution $P$ is typically concentrated:

$$
\sum_{j \in \text{top-}k} P_{ij} \gg 1 - \epsilon, \quad k \ll L
\tag{3.4}
$$

**Empirical evidence.** **Figure 3.1** shows the cumulative attention mass 
vs ranked keys for a middle GPT-2 layer. Top 10% of keys capture 
**{{TOP10_MASS}}%** of mass; top 50% capture **{{TOP50_MASS}}%**.

**Implication.** Pruning to the top $k = (1 - \rho) L$ keys retains most 
of the output quality, at fraction $1 - \rho$ of the compute.

## 3.3 Top-$k$ Selection

For a score vector $s \in \mathbb{R}^L$, the top-$k$ selection produces:

$$
\text{top-}k(s)_j = \begin{cases} s_j & \text{if } s_j \geq \tau \\ -\infty & \text{otherwise} \end{cases}
\tag{3.5}
$$

where $\tau$ is the $k$-th largest value.

**Non-differentiability.** The gradient $\partial \text{top-}k / \partial s$ 
is zero almost everywhere. This is problematic for end-to-end training.

## 3.4 Soft Top-$k$ Relaxations

**Sigmoid mask.** Given $\tau$, we can define:

$$
m_j = \sigma\!\left(\frac{s_j - \tau}{T}\right) \in (0, 1)
\tag{3.6}
$$

so that $m_j \to 1$ for $s_j > \tau$ and $m_j \to 0$ for $s_j < \tau$. 
As $T \to 0$, $m_j \to \mathbb{1}[s_j > \tau]$, recovering hard top-$k$.

**Application in log space.** Instead of a multiplicative mask, we add a 
log-mask to the scores:

$$
\tilde{s}_j = s_j + \log(m_j + \epsilon)
\tag{3.7}
$$

This preserves the softmax interpretation: $\text{softmax}(\tilde{s}) = 
\text{softmax}(s) \cdot m / \sum m$.

## 3.5 Per-Head Sparsity Budgets

The retention budget $k_h$ for head $h$ is:

$$
k_h = \lfloor (1 - \rho_h) L \rfloor
\tag{3.8}
$$

where $\rho_h \in [0, 1]$ is the head-specific sparsity ratio.

**Learning $\rho_h$.** We parameterize via a scalar logit:

$$
\rho_h = \sigma(\ell_h), \qquad \ell_h \in \mathbb{R}
\tag{3.9}
$$

This is differentiable w.r.t. $\ell_h$, allowing gradient-based learning of 
the sparsity budget.

## 3.6 Chunked Computation

To bound peak memory, we process the query dimension in chunks of size $C$:

$$
O_{[i:i+C]} = \text{softmax}\!\left(\frac{Q_{[i:i+C]} K^\top}{\sqrt{d_h}}\right) V
\tag{3.10}
$$

**Peak memory.** $\mathcal{O}(B H C L)$ instead of $\mathcal{O}(B H L^2)$.

For $C = 64$, $B = 8$, $H = 12$, $L = 512$: $64 \times 12 \times 64 \times 512 \times 2 = 50$ MB 
instead of $8 \times 12 \times 512^2 \times 2 = 400$ MB. **8× reduction.**

## 3.7 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.11}
$$

with $B = 0$ at initialization. ASA uses LoRA for recovery after sparsity 
injection.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $S$ | Score matrix |
| $P$ | Attention probabilities |
| $\rho_h$ | Sparsity ratio for head $h$ |
| $\ell_h$ | Learnable logit for head $h$ |
| $k_h$ | Retention budget |
| $\tau_h$ | Threshold |
| $T$ | Temperature |
| $C$ | Chunk size |
