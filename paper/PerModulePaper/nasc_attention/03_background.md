<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

Cost: $\mathcal{O}(L^2 d)$.

## 3.2 Sliding-Window Attention

$$
M_{ij} = \mathbf{1}[|i - j| \leq W]
\tag{3.2}
$$

Cost: $\mathcal{O}(L W d)$.

## 3.3 Sparse Block Attention

Partition sequence into $n_b = L / B_s$ blocks. Compute block-level scores:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}, \quad \bar{q}_i = \text{mean}(Q_{\text{block } i})
\tag{3.3}
$$

Select top-$k$ blocks per query block:

$$
\mathcal{B}_i = \text{topk}_{k}(\{s_{ij}\}_j)
\tag{3.4}
$$

Then compute full attention within selected blocks.

## 3.4 Random Fourier Features

Approximate $\exp(q^\top k)$ via:

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k)
\tag{3.5}
$$

with $\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)]$ where 
$\Omega \in \mathbb{R}^{d \times F}$ is sampled from a Gaussian.

**Cost:** $O(L F d)$ per head, linear in $L$.

## 3.5 Softpick

Softpick (Alman & Song, 2024) replaces softmax with:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{x_i - m} - e^{-m})}{\sum_j \text{ReLU}(e^{x_j - m} - e^{-m})}
\tag{3.6}
$$

where $m = \max_j x_j$.

**Issues.** When all $x_j$ are negative (masked rows), $e^{-m}$ dominates 
and the denominator can vanish. NASC handles this with a **batch-level 
fallback** to standard softmax.

## 3.6 LoRA

Standard low-rank adaptation (Hu et al., 2021).

## 3.7 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of heads |
| $W$ | Window size |
| $B_s$ | Block size |
| $k$ | Top-k blocks |
| $F$ | RFF features |
| $m$ | Row maximum (softpick) |
</parameter>

---