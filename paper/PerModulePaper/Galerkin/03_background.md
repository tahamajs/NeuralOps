<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Scaled Dot-Product Attention

For input $X \in \mathbb{R}^{B \times N \times d}$, attention computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
\tag{3.1}
$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$.

**Cost.** Time and memory for $QK^\top$ are $\mathcal{O}(N^2)$.

## 3.2 Discrete Cosine Transform (DCT)

The DCT-II basis is defined as:

$$
\Phi_{n, k} = \cos\!\left(\frac{\pi k (2n + 1)}{2N}\right), \quad
n = 0, \dots, N-1, \quad k = 0, \dots, N-1
\tag{3.2}
$$

with normalization:

$$
\Phi_{n, 0} \to \frac{1}{\sqrt{N}} \Phi_{n, 0}, \quad
\Phi_{n, k} \to \sqrt{\frac{2}{N}} \Phi_{n, k} \text{ for } k \geq 1
\tag{3.3}
$$

The full DCT basis $\Phi \in \mathbb{R}^{N \times N}$ is orthonormal: 
$\Phi^\top \Phi = I_N$.

**Why DCT?** DCT is optimal for energy compaction of smooth signals 
(Ahmed et al., 1974). For smooth attention kernels, DCT projection preserves 
most energy.

**Truncation.** Keeping only the first $r$ columns gives 
$\Phi_r \in \mathbb{R}^{N \times r}$, which satisfies 
$\Phi_r^\top \Phi_r = I_r$.

## 3.3 Petrov-Galerkin Projection

Consider the continuous integral operator:

$$
(\mathcal{K} f)(x) = \int_0^1 k(x, y) f(y) \, dy
\tag{3.4}
$$

Let $\{\phi_1, \dots, \phi_r\}$ be an orthonormal basis of a subspace 
$\mathcal{V}_r \subset L^2([0, 1])$. The Galerkin approximation is:

$$
\mathcal{K}_r = \sum_{i, j = 1}^r K_{ij} \phi_i \otimes \phi_j
\tag{3.5}
$$

where $K_{ij} = \langle \phi_i, \mathcal{K} \phi_j \rangle$.

**Discretization.** For a discrete sequence of length $N$, the Galerkin 
approximation becomes:

$$
\mathcal{K}_r = \Phi_r \, K_r \, \Phi_r^\top, \quad
K_r \in \mathbb{R}^{r \times r}
\tag{3.6}
$$

with $\Phi_r \in \mathbb{R}^{N \times r}$ being the discrete basis.

## 3.4 LoRA Recovery

After replacing attention, we freeze base weights and train LoRA adapters:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.7}
$$

with $A \in \mathbb{R}^{r_L \times k}$, $B \in \mathbb{R}^{d \times r_L}$, 
$B = 0$ at init.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $N$ | Sequence length |
| $d$ | Hidden dimension |
| $H$ | Number of heads |
| $r$ | Basis dimension |
| $\Phi$ | Orthonormal basis ($N \times r$) |
| $\Phi_r$ | Truncated basis |
| $r_L$ | LoRA rank |
| $\mathcal{K}$ | Integral operator |
| $k(x, y)$ | Kernel function |
```

---