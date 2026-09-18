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
S = Q K^\top, \qquad
P = \text{softmax}\!\left(\frac{S}{\sqrt{d_h}}\right), \qquad
\text{out} = P V
\tag{3.1}
$$

with $Q, K, V$ projected from $X$. The score matrix $S$ has shape 
$B \times H \times L \times L$ and requires $\mathcal{O}(L^2 d_h)$ FLOPs.

## 3.2 Johnson–Lindenstrauss Lemma

**Lemma (JL, 1984).** For any $0 < \epsilon < 1$ and any set of $N$ points 
$\{x_i\}_{i=1}^N \subset \mathbb{R}^d$, there exists a linear map 
$\Omega : \mathbb{R}^d \to \mathbb{R}^r$ with 
$r = \mathcal{O}(\log N / \epsilon^2)$ such that:

$$
(1 - \epsilon) \|x_i - x_j\|_2^2 \leq \|\Omega x_i - \Omega x_j\|_2^2 \leq (1 + \epsilon) \|x_i - x_j\|_2^2
\tag{3.2}
$$

**Gaussian construction.** A common choice is $\Omega$ with i.i.d. 
$\mathcal{N}(0, 1/r)$ entries. Then:

$$
\mathbb{E}[\|\Omega x\|_2^2] = \|x\|_2^2
\tag{3.3}
$$

and inner products are preserved:

$$
\mathbb{E}[(x_i \Omega)(x_j \Omega)^\top] = x_i x_j^\top
\tag{3.4}
$$

## 3.3 Sketching and Inner Products

By the JL lemma, if $\Omega$ is a Gaussian matrix with variance $1/r$, then:

$$
|\tilde{Q}_i \tilde{K}_j^\top - Q_i K_j^\top| \leq \epsilon \|Q_i\|_2 \|K_j\|_2
\tag{3.5}
$$

with probability $1 - \delta$ for $r \geq \mathcal{O}(\log(1/\delta) / \epsilon^2)$.

**In RPC.** We use $r = 32$ (much smaller than $d_h = 64$) and scale 
$\Omega$ by $d_h^{-1/2}$. The result approximates $Q_i K_j^\top$ with 
controllable error.

## 3.4 Scale Factor

Standard attention scales by $1/\sqrt{d_h}$ to prevent softmax saturation. 
RPC scales by $1/\sqrt{r}$:

$$
s_{ij} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}
\tag{3.6}
$$

**Rationale.** The sketched $\tilde{Q}_i$ has variance $\|Q_i\|_2^2$ 
(matched to $Q_i$), and similarly for $\tilde{K}_j$. If the sketch preserves 
inner products, then $\tilde{Q}_i \tilde{K}_j^\top$ has the same scale as 
$Q_i K_j^\top$. We divide by $\sqrt{d_h}$ to match attention, but the code 
uses $\sqrt{r}$ — a subtle choice.

## 3.5 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Head count |
| $r$ | Sketch dimension |
| $\Omega$ | Random projection |
| $\tilde{Q}, \tilde{K}$ | Sketched Q, K |
| $\epsilon$ | JL distortion |
| $r_L$ | LoRA rank |

---