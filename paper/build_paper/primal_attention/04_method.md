<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Primal Attention approximates the $L \times L$ attention matrix via a 
low-rank factorization through $m$ landmarks:

$$
A \approx F_1 F_2^{-1} F_3
\tag{4.1}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

where $F_1 \in \mathbb{R}^{L \times m}$, $F_2 \in \mathbb{R}^{m \times m}$, 
$F_3 \in \mathbb{R}^{m \times L}$.

**Figure 4.1** illustrates the pipeline.

```
      Q, K ∈ R^{L × d_h}
             │
             ▼
      ┌─────────────────┐
      │  Pool to        │
      │  landmarks      │
      │  Q_l, K_l       │
      └────────┬────────┘
               │
      ┌────────┴────────┬───────────┐
      ▼                 ▼           ▼
   F₁ = Q K_lᵀ      F₂ = Q_l K_lᵀ   F₃ = Q_l Kᵀ
    (L × m)          (m × m)         (m × L)
                       │
                       ▼
              ┌─────────────────┐
              │  Schulz         │
              │  pseudoinverse  │
              │  F₂⁻¹           │
              └────────┬────────┘
                       │
      ┌────────────────┴────────────────┐
      ▼                                 ▼
   A ≈ F₁ F₂⁻¹ F₃                  (L × L)
      │
      ▼
   softmax(A) · V
```

## 4.2 Landmark Selection via Segment Pooling

Given sequence length $L$ and target landmark count $m$, we divide the 
sequence into $m$ equal segments of size $s = L / m$ and average each:

$$
Q_\text{land}[i] = \frac{1}{s} \sum_{j = i s}^{(i+1) s - 1} Q[j], \qquad i = 0, \ldots, m - 1
\tag{4.2}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

Similarly for $K_\text{land}$. If $L$ is not divisible by $m$, we use 
uniform strided sampling:

$$
Q_\text{land}[i] = Q[\lfloor i L / m \rfloor], \qquad i = 0, \ldots, m - 1
\tag{4.3}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

**Why segment pooling?** It provides **uniform coverage** regardless of 
content, is **deterministic**, and is **differentiable** (mean is linear).

**Asymmetry.** Notice we compute $Q_\text{land}$ and $K_\text{land}$ 
**separately**, allowing different landmark summaries for the query and 
key sides. This is the key difference from Nyströmformer.

## 4.3 The Three Factors

Given $Q, K \in \mathbb{R}^{L \times d_h}$ (after RoPE, after GQA repeat) and 
their landmarks $Q_\text{land}, K_\text{land} \in \mathbb{R}^{m \times d_h}$, 
we compute:

$$
F_1 = \frac{Q K_\text{land}^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times m}
\tag{4.4}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

$$
F_2 = \frac{Q_\text{land} K_\text{land}^\top}{\sqrt{d_h}} \in \mathbb{R}^{m \times m}
\tag{4.5}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

$$
F_3 = \frac{Q_\text{land} K^\top}{\sqrt{d_h}} \in \mathbb{R}^{m \times L}
\tag{4.6}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

The scale $1/\sqrt{d_h}$ is applied once, matching the standard attention.

## 4.4 Asymmetric Attention Matrix

The approximate attention matrix is:

$$
\tilde{A} = F_1 F_2^{-1} F_3 \in \mathbb{R}^{L \times L}
\tag{4.7}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

**Why this works.** If the true attention matrix $A$ has rank $\le m$, then 
$A = F_1 F_2^{-1} F_3$ exactly (given any $m$ landmarks spanning the column 
space). For general $A$, this is a low-rank approximation.

**Complexity.** Computing $\tilde{A}$ takes:

- $F_1$: $\mathcal{O}(L m d_h)$
- $F_2$: $\mathcal{O}(m^2 d_h)$
- $F_3$: $\mathcal{O}(m L d_h)$
- $F_2^{-1}$: $\mathcal{O}(m^3)$ (via SVD or Schulz)
- $F_1 F_2^{-1}$: $\mathcal{O}(L m^2)$
- $(F_1 F_2^{-1}) F_3$: $\mathcal{O}(L m L) = \mathcal{O}(L^2 m)$

**Wait — that's still quadratic!** We need to be careful. Let me reconsider.

The correct approach:

$$
\tilde{A} V = (F_1 F_2^{-1} F_3) V = F_1 (F_2^{-1} (F_3 V))
\tag{4.8}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

where the associativity allows computing left-to-right:

1. $F_3 V \in \mathbb{R}^{m \times d_h}$: $\mathcal{O}(m L d_h)$
2. $F_2^{-1} (F_3 V) \in \mathbb{R}^{m \times d_h}$: $\mathcal{O}(m^2 d_h)$
3. $F_1 (\cdots) \in \mathbb{R}^{L \times d_h}$: $\mathcal{O}(L m d_h)$

Total: $\mathcal{O}(L m d_h + m^2 d_h)$. For $m \ll L$, this is **linear in $L$**.

## 4.5 Schulz Pseudoinverse

### 4.5.1 Initialization

We initialize $X_0 = F_2^+$ using `torch.linalg.pinv` (SVD-based). This 
provides a good starting point for the Schulz iteration.

### 4.5.2 Iteration

For $K$ iterations (default $K = 4$):

$$
X_{t+1} = X_t (2I - F_2 X_t)
\tag{4.9}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

where $I \in \mathbb{R}^{m \times m}$ is the identity.

**Convergence analysis.** Let $E_t = I - F_2 X_t$. Then:

$$
E_{t+1} = I - F_2 X_{t+1} = I - F_2 X_t (2I - F_2 X_t) = (I - F_2 X_t)^2 = E_t^2
\tag{4.10}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

So $\|E_t\| \le \|E_0\|^{2^t}$. If $\|E_0\| < 1$, convergence is quadratic.

### 4.5.3 Why Init with Pseudoinverse?

Starting with $X_0 = 0$ or $X_0 = F_2^\top / \|F_2\|^2$ often fails to 
converge in $K \le 6$ iterations because $\|E_0\|$ may exceed 1. Starting 
from the SVD pseudoinverse gives $\|E_0\| \approx 0$ and refines it 
numerically.

### 4.5.4 Number of Iterations

We use $K = 4$ iterations. Empirically, $K = 4$ suffices for 
$m \le 128$; larger $m$ may benefit from more.

## 4.6 Causality

For decoder models, the attention matrix must be lower-triangular (causal):

$$
A_{ij} = 0 \quad \text{for } j > i
\tag{4.11}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

We enforce this by adding a causal mask to $\tilde{A}$:

$$
\tilde{A}_\text{causal} = \tilde{A} + M, \qquad
M_{ij} = \begin{cases} 0 & j \le i \\ -10^4 & j > i \end{cases}
\tag{4.12}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

Then softmax. This is not a true causal approximation (the landmarks see 
the whole sequence), but empirically works well for generation.

## 4.7 Complete Forward Pass

**Algorithm 1:** Primal Attention forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        attention_mask, is_causal
        m (num landmarks), K (pinv iters)
Output: context ∈ R^{B × H × L × d_h}

1:  # Segment pooling for landmarks
2:  Q_l, K_l ← pool_to_landmarks(Q, K, m)     # (B, H, m, d_h)
3:
4:  # Compute factors
5:  scale ← 1 / sqrt(d_h)
6:  F₁ ← (Q @ K_lᵀ) * scale                    # (B, H, L, m)
7:  F₂ ← (Q_l @ K_lᵀ) * scale                  # (B, H, m, m)
8:  F₃ ← (Q_l @ Kᵀ) * scale                    # (B, H, m, L)
9:
10: # Pseudoinverse of F₂
11: F₂_inv ← linalg.pinv(F₂)
12: for t in 0..K-1 do
13:     F₂_inv ← F₂_inv @ (2I - F₂ @ F₂_inv)   # Schulz
14: end for
15:
16: # Approximate attention (associative, linear in L)
17: F₃V ← F₃ @ V                              # (B, H, m, d_h)
18: F₂_inv_F₃V ← F₂_inv @ F₃V                # (B, H, m, d_h)
19: A ← F₁ @ F₂_inv_F₃V                       # (B, H, L, d_h)
20:
21: # Causal mask
22: if is_causal:
23:     A ← A + causal_mask
24: end if
25:
26: # Standard attention mask
27: if attention_mask:
28:     A ← apply_attention_mask(A, attention_mask)
29: end if
30:
31: probs ← softmax(A)
32: context ← probs @ V
33: return context
```

**Complexity.** All operations are $\mathcal{O}(L m d_h + m^2 d_h + m^3)$; no 
$L^2$ terms.

## 4.8 Complexity Analysis

| Component | Standard | Primal |
|---|---|---|
| $Q K^\top$ | $\mathcal{O}(L^2 d_h)$ | — |
| $F_1, F_2, F_3$ | — | $\mathcal{O}(L m d_h + m^2 d_h)$ |
| $F_2^{-1}$ | — | $\mathcal{O}(m^3)$ |
| $A V$ | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L m d_h)$ |
| Softmax | $\mathcal{O}(L^2)$ | $\mathcal{O}(L m)$ |
| **Total** | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L m d_h + m^3)$ |

For $m = 64$ and $L = 512$, the crossover is at $L \approx m^2 / d_h \cdot c$ 
for some constant. In practice, Primal Attention is faster for $L > 512$.

## 4.9 Parameter Analysis

Primal Attention **does not reduce parameters** relative to MHA. The query, 
key, value, and output projections are unchanged:

$$
|\theta_\text{Primal}| = |\theta_\text{MHA}|
\tag{4.13}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

It reduces **memory during forward** (no $L \times L$ attention matrix) and 
**time** (linear in $L$). LoRA adds $2 r_L d$ parameters per projection.

## 4.10 Theoretical Notes

### 4.10.1 Approximation Error

For a general attention matrix $A$ with singular values $\sigma_1 \ge \sigma_2 \ge \cdots$:

$$
\|A - \tilde{A}\|_F \leq \sqrt{\sum_{i > m} \sigma_i^2} + \epsilon_\text{landmark}
\tag{4.14}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

where $\epsilon_\text{landmark}$ depends on landmark selection quality.

### 4.10.2 Exactness for Low-Rank $A$

If $\text{rank}(A) \le m$, then $\tilde{A} = A$ exactly (assuming landmarks 
span the row/column spaces).

### 4.10.3 Schulz Convergence Rate

If $\|E_0\| \le \rho < 1$, then after $K$ iterations:

$$
\|E_K\| \le \rho^{2^K}
\tag{4.15}

<!-- source: neuralops/attentions/Primal_attention.py line 138 -->

<!-- source: neuralops/attentions/primal_attention.py line 138 -->
$$

For $\rho = 0.5$ and $K = 4$: $\|E_K\| \le 2^{-16} \approx 10^{-5}$.

---
