<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Softmax Attention

For $Q, K, V \in \mathbb{R}^{N \times d_h}$:

$$
A = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times N}, \qquad
\text{Output} = A V \in \mathbb{R}^{N \times d_h}
\tag{3.1}
$$

Cost: $\mathcal{O}(N^2 d_h)$ time and $\mathcal{O}(N^2)$ memory.

## 3.2 The Nyström Method

For a symmetric positive semi-definite kernel matrix $K \in \mathbb{R}^{N \times N}$, 
choose $m$ landmark indices $\{i_1, \dots, i_m\}$. Then:

$$
K \approx K_{:, m} K_{m, m}^{-1} K_{m, :}
\tag{3.2}
$$

where:

- $K_{:, m} \in \mathbb{R}^{N \times m}$: columns at landmarks
- $K_{m, m} \in \mathbb{R}^{m \times m}$: submatrix at landmarks
- $K_{m, :} \in \mathbb{R}^{m \times N}$: rows at landmarks

The inverse $K_{m, m}^{-1}$ exists iff $K_{m, m}$ is invertible. For 
numerical stability, use the **pseudoinverse** $K_{m, m}^+$.

**Approximation error.** If $K$ has rank exactly $m$, the approximation is 
exact. In practice, $K$ has low effective rank, making the approximation 
useful.

## 3.3 Nyström Attention

The softmax attention matrix $A = \mathrm{softmax}(Q K^\top / \sqrt{d_h})$ 
is not symmetric, but the Nyström structure still applies:

$$
\tilde{A} = \underbrace{\mathrm{softmax}(Q K_m^\top / \sqrt{d_h})}_{N \times m} \cdot \underbrace{\mathrm{pinv}(\mathrm{softmax}(Q_m K_m^\top / \sqrt{d_h}))}_{m \times m} \cdot \underbrace{\mathrm{softmax}(Q_m K^\top / \sqrt{d_h})}_{m \times N}
\tag{3.3}
$$

Then output:

$$
\text{Output} = \tilde{A} V = \mathrm{softmax}(Q K_m^\top) \cdot \left[ \mathrm{pinv}(\mathrm{softmax}(Q_m K_m^\top)) \cdot \mathrm{softmax}(Q_m K^\top) V \right]
\tag{3.4}
$$

**Associative multiplication.** The brackets can be computed left-to-right:

1. $\mathrm{softmax}(Q_m K^\top) V \in \mathbb{R}^{m \times d_h}$ — cost $\mathcal{O}(N m d_h)$
2. $\mathrm{pinv}(A_{mm}) \cdot \text{step1} \in \mathbb{R}^{m \times d_h}$ — cost $\mathcal{O}(m^2 d_h)$
3. $\mathrm{softmax}(Q K_m^\top) \cdot \text{step2} \in \mathbb{R}^{N \times d_h}$ — cost $\mathcal{O}(N m d_h)$

Total: $\mathcal{O}(N m d_h)$ instead of $\mathcal{O}(N^2 d_h)$.

## 3.4 Landmark Selection by Adaptive Pooling

Nyströmformer uses `adaptive_avg_pool1d` on $Q, K$:

$$
Q_m = \mathrm{AdaptiveAvgPool1d}(Q, m) \in \mathbb{R}^{m \times d_h}
\tag{3.5}
$$

This partitions the sequence into $m$ contiguous segments and averages 
each. It is:

- **Deterministic** — no randomness.
- **Efficient** — $\mathcal{O}(N d_h)$.
- **Locality-preserving** — adjacent tokens share landmarks.

## 3.5 Iterative Pseudoinverse (Newton-Schulz)

For a matrix $A$, the Moore-Penrose pseudoinverse $A^+$ satisfies:

$$
A A^+ A = A, \quad A^+ A A^+ = A^+, \quad (A A^+)^\top = A A^+, \quad (A^+ A)^\top = A^+ A
\tag{3.6}
$$

The Newton-Schulz iteration:

$$
V_{k+1} = V_k (2 I - A V_k), \quad V_0 = \alpha A^\top
\tag{3.7}
$$

converges to $A^+$ if $\|I - \alpha A A^\top\|_2 < 1$. Preconditioning 
with $\alpha = 1 / (\|A\|_1 \|A\|_\infty + \epsilon)$ ensures this.

**Convergence rate.** Quadratic — each iteration roughly squares the error.

**Iterations.** 3 iterations suffice for most matrices.

## 3.6 Causal Masking

For causal decoders, the attention matrix is lower-triangular:

$$
A_{ij} = \begin{cases} \text{valid} & i \geq j \\ 0 & i < j \end{cases}
\tag{3.8}
$$

Nyström approximation must respect this. **Nyströmformer is not naturally 
causal** — the landmark pooling averages across all tokens, including 
future ones. Therefore, we fall back to dense causal attention for decoders.

## 3.7 Multi-Head Self-Attention

Standard MHSA:

$$
\text{MHSA}(X) = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_H) W_O
\tag{3.9}
$$

Nyströmformer applies the approximation per head.

## 3.8 Grouped-Query Attention (GQA)

$H_{kv} < H$ KV heads, repeated $H / H_{kv}$ times via `repeat_kv`.

## 3.9 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) rotates $Q, K$ by position-dependent angles.

## 3.10 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable residuals:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{3.10}
$$

## 3.11 Notation Summary

| Symbol | Meaning |
|---|---|
| $N$ | Sequence length |
| $m$ | Number of landmarks |
| $Q, K, V$ | Query, Key, Value |
| $Q_m, K_m$ | Landmark queries, keys |
| $A_{mm}$ | Landmark attention matrix |
| $\text{pinv}(\cdot)$ | Moore-Penrose pseudoinverse |
| $\alpha$ | Pinv preconditioning scalar |
| $d_h, H, H_{kv}$ | Model dims |

---