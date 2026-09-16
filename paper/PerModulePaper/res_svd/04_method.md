# 4. Method

## 4.1 Overview

Res-SVD Attention replaces each projection with a two-level decomposition. 
**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────┐
    │  Original weight W ∈ R^{m×n}         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 1: SVD of W                    │
    │  W = U Σ V^T                         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 2: Base = top r_b              │
    │  W_base = U_b Σ_b V_b^T              │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 3: Residual = W - W_base       │
    │  R = W - W_base                      │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 4: SVD of R                    │
    │  R = U_r Σ_r V_r^T                   │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Step 5: Init LoRA from residual     │
    │  B = U_r Σ_r^{1/2}, A = Σ_r^{1/2} V_r^T │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Frozen: U_b, V_b                    │
    │  Trainable: B, A                     │
    └──────────────────────────────────────┘
```

## 4.2 Two-Level Decomposition

### 4.2.1 Base SVD

Given $W \in \mathbb{R}^{m \times n}$:
$$
U, \Sigma, V^\top = \text{SVD}(W)
\tag{4.1}
$$

Take top $r_b$:
$$
W_b = U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top
\tag{4.2}
$$

Store $U_b = U_{:, :r_b}$, $V_b = V_{:r_b, :}$ as **frozen** parameters.

### 4.2.2 Residual SVD

$$
R = W - W_b
\tag{4.3}
$$

$$
U_r, \Sigma_r, V_r^\top = \text{SVD}(R)
\tag{4.4}
$$

Take top $r_r$:
$$
R_r = U_{r, :, :r_r} \Sigma_{r, :r_r} V_{r, :r_r, :}^\top
\tag{4.5}
$$

### 4.2.3 LoRA Initialization

Split $\Sigma_r^{1/2}$ symmetrically:
$$
B = U_{r, :, :r_r} \Sigma_{r, :r_r}^{1/2}, \quad
A = \Sigma_{r, :r_r}^{1/2} V_{r, :r_r, :}^\top
\tag{4.6}
$$

Store $B \in \mathbb{R}^{m \times r_r}$, $A \in \mathbb{R}^{r_r \times n}$ as 
**trainable**.

## 4.3 Forward Pass

$$
y = x (U_b V_b)^\top + \alpha x (B A)^\top + b
\tag{4.7}
$$

where $\alpha$ is a scaling factor (default 1.0).

**Alternative factorization for efficiency:**
```python
# Base
h_b = x @ V_b.t()              # (B, L, r_b)
y_b = h_b @ U_b.t()            # (B, L, m)

# Residual
h_r = x @ A.t()                 # (B, L, r_r)
y_r = h_r @ B.t()               # (B, L, m)

y = y_b + alpha * y_r + b
```

## 4.4 Parameter Count

| Component | Parameters |
|---|---|
| Base SVD | $r_b(m + n + 1)$ |
| Residual LoRA | $r_r(m + n)$ |
| Bias | $m$ |
| **Total** | $(r_b + r_r)(m + n) + r_b + m$ |

**Baseline:** $mn$.

For $m = n = 768$, $r_b = 16$, $r_r = 4$:

| Component | Params |
|---|---|
| Base SVD | $16 \times 1537 = 24,592$ |
| Residual LoRA | $4 \times 1536 = 6,144$ |
| Bias | 768 |
| **Total** | **31,504** |
| Baseline | 589,824 |
| **Ratio** | **~18.7×** |

## 4.5 Full Algorithm

**Algorithm 1:** Res-SVD factorization
```
Input:  Weight W ∈ R^{m×n}, ranks r_b, r_r, scaling α
Output: Base factors (U_b, V_b), Residual factors (B, A), bias b

1:  U, Σ, V^T ← SVD(W)
2:  # Base
3:  U_b ← U[:, :r_b];  Σ_b ← Σ[:r_b];  V_b^T ← V^T[:r_b, :]
4:  W_base ← U_b @ diag(Σ_b) @ V_b^T
5:  # Residual
6:  R ← W - W_base
7:  U_r, Σ_r, V_r^T ← SVD(R)
8:  # LoRA init
9:  Σ_r_sqrt ← sqrt(Σ_r[:r_r])
10: B ← U_r[:, :r_r] * Σ_r_sqrt[None, :]
11: A ← Σ_r_sqrt[:, None] * V_r^T[:r_r, :]
12: return (U_b, V_b, B, A, b)
```

**Algorithm 2:** Res-SVD forward pass
```
Input:  x ∈ R^{B×L×n}, base (U_b, V_b), residual (B, A), bias b, α
Output: y ∈ R^{B×L×m}

1: h_b ← x @ V_b^T
2: y_b ← h_b @ U_b^T
3: h_r ← x @ A^T
4: y_r ← h_r @ B^T
5: y ← y_b + α · y_r + b
6: return y
```

## 4.6 Complexity Analysis

| Component | Time | Space |
|---|---|---|
| Base forward | $\mathcal{O}(L n r_b)$ | $\mathcal{O}(L r_b)$ |
| Residual forward | $\mathcal{O}(L n r_r)$ | $\mathcal{O}(L r_r)$ |
| Total | $\mathcal{O}(L n (r_b + r_r))$ | $\mathcal{O}(L (r_b + r_r))$ |

**Baseline:** $\mathcal{O}(L n m)$.

For $m = n = 768$, $r_b + r_r = 20$: **~38× reduction in projection FLOPs**.

## 4.7 Theoretical Notes

### 4.7.1 Equivalence to Rank-$(r_b + r_r)$ SVD

**Claim.** Res-SVD with $(r_b, r_r)$ is mathematically equivalent to 
truncated SVD of rank $r_b + r_r$.

**Proof.** The columns of $U$ are orthogonal, so:
$$
U_{:, :r_b} \Sigma_{:r_b} V_{:r_b, :}^\top + U_{:, r_b:r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top = U_{:, :r_b+r_r} \Sigma_{:r_b+r_r} V_{:r_b+r_r, :}^\top
$$
$\square$

**Implication:** Res-SVD does not offer a better approximation than 
rank-$(r_b + r_r)$ SVD. Its advantage is **PEFT structure** — the base 
is frozen and the residual is trainable.

### 4.7.2 Optimality at Initialization

**Claim.** At initialization, the combined Res-SVD approximation has 
reconstruction error $\|W - W_b - R_r\|_F = \sqrt{\sum_{i > r_b+r_r} \sigma_i^2}$.

**Proof.** Direct consequence of 4.7.1. $\square$

### 4.7.3 Gradient Flow

**Claim.** Only the residual $B, A$ receives gradient during training.

**Proof.** $U_b, V_b$ have `requires_grad = False`. Only $B, A$ are updated. $\square$

### 4.7.4 Causality

**Proof.** Position-wise projections + causal attention mask. $\square$

---