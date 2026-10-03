<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 The CUR Decomposition

For a matrix $W \in \mathbb{R}^{m \times n}$ and target rank $r$, CUR selects:

$$
W \approx C \cdot U^+ \cdot R
\tag{3.1}
$$

where:
- $C \in \mathbb{R}^{m \times r}$ contains $r$ columns of $W$.
- $R \in \mathbb{R}^{r \times n}$ contains $r$ rows of $W$.
- $U \in \mathbb{R}^{r \times r}$ is the intersection matrix $W[\text{rows}, \text{cols}]$.
- $U^+$ is the Moore–Penrose pseudoinverse.

**Error bound.** Drineas et al. (2008) prove that if columns and rows are 
sampled with probability proportional to leverage scores, then:

$$
\|W - C U^+ R\|_F \leq (1 + \epsilon) \|W - W_r\|_F
\tag{3.2}
$$

with probability $1 - \delta$ for sample size 
$r = \mathcal{O}(r^* / \epsilon^2 \log(1/\delta))$.

## 3.2 Leverage Scores

For a matrix $W = U \Sigma V^\top$ with rank $r^*$, the **column leverage score** 
of column $j$ is:

$$
\pi_j = \sum_{k=1}^{r^*} (V_{j, k})^2
\tag{3.3}
$$

and the **row leverage score** of row $i$ is:

$$
\rho_i = \sum_{k=1}^{r^*} (U_{i, k})^2
\tag{3.4}
$$

Leverage scores satisfy $\sum_j \pi_j = r^*$ and $\sum_i \rho_i = r^*$. The 
scores measure how important each row/column is for the low-rank structure 
of $W$.

**Deterministic top-$r$ selection.** Instead of sampling, PHANTOM selects 
the $r$ rows/columns with highest leverage. This is deterministic and gives 
the best rank-$r$ CUR approximation in practice.

## 3.3 Angular LSH

Angular LSH (Charikar, 2002) hashes vectors based on their direction. For a 
random projection $a \sim \mathcal{N}(0, I_d)$, define:

$$
h_a(x) = \text{sign}(a^\top x) \in \{-1, +1\}
\tag{3.5}
$$

For $k$ independent projections $a_1, \dots, a_k$, the $k$-bit hash is:

$$
H(x) = \sum_{i=1}^{k} 2^{i-1} \cdot \mathbb{1}[h_{a_i}(x) = +1]
\tag{3.6}
$$

**Collision probability.** For two vectors $x, y$ with angle $\theta$ between 
them:

$$
\Pr[H(x) = H(y)] = 1 - \frac{\theta}{\pi}
\tag{3.7}
$$

so vectors with small angles collide with high probability.

**Multi-round LSH.** A single LSH round has high variance. PHANTOM applies 
$R_\ell$ independent rounds, each with a fresh projection matrix, and 
averages the resulting attention outputs:

$$
\text{Output} = \frac{1}{R_\ell} \sum_{r=1}^{R_\ell} \text{Attn}_{\text{LSH}}^{(r)}(Q, K, V)
\tag{3.8}
$$

## 3.4 Softpick Activation

Softpick (Lu et al., 2024) generalizes softmax by subtracting a fixed baseline:

$$
\text{softpick}(x_i) = \text{ReLU}\!\left(e^{(x_i - m)/\tau_s} - e^{-m/\tau_s}\right)
\tag{3.9}
$$

where $m = \max_j x_j$ and $\tau_s$ is a temperature. Positions with 
$x_i \ll m$ are truncated to zero, producing sparse attention.

**Normalization.** The unnormalized weights are normalized:

$$
\hat{p}_i = \frac{\text{softpick}(x_i)}{\sum_j \text{softpick}(x_j) + \epsilon}
\tag{3.10}
$$

**Numerical issue.** If $m < 0$ (all attention scores negative, e.g., from 
masking), then $e^{-m/\tau_s} \geq 1$ and the ReLU may zero out the entire 
row. PHANTOM adds a softmax fallback:

$$
p_i = \begin{cases} \hat{p}_i & \text{if } \sum_j \text{softpick}(x_j) > 10^{-5} \\ \text{softmax}(x_i / \tau_s) & \text{otherwise} \end{cases}
\tag{3.11}
$$

## 3.5 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters to frozen base weights:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.12}
$$

PHANTOM uses LoRA on top of the CUR-factorized projections for recovery 
fine-tuning.

## 3.6 Notation Summary

| Symbol | Meaning |
|---|---|
| $W$ | Weight matrix |
| $U, \Sigma, V$ | SVD factors |
| $C, R, U^+$ | CUR factors |
| $\pi_j$ | Column leverage |
| $\rho_i$ | Row leverage |
| $r$ | Target rank |
| $R_\ell$ | LSH rounds |
| $k$ | LSH bits |
| $\tau_s$ | Softpick temperature |
| $B$ | Bucket size |
| $L$ | Sequence length |

---