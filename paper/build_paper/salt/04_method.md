<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

SALT replaces each linear projection $W$ with a three-component parameterization:

```
    Weight W ∈ R^{d_out × d_in}
            │
            ▼
    ┌─────────────────────────┐
    │  Truncated SVD          │
    │  W ≈ U_r Σ_r V_r^T      │
    └───────────┬─────────────┘
                │
    ┌───────────┴────────────┐
    ▼                        ▼
    ┌──────────────┐   ┌─────────────────┐
    │  Buffers     │   │  Trainable      │
    │  U, S, Vt    │   │  γ, A, B        │
    │  (frozen)    │   │                 │
    └──────────────┘   └─────────────────┘
                │                 │
                └────────┬────────┘
                         ▼
    ┌──────────────────────────────────────┐
    │  Forward:                             │
    │  W_eff = U diag(S ⊙ γ) V^T + A B     │
    │  y = W_eff x + b                      │
    └──────────────────────────────────────┘
```

## 4.2 The SVD Skeleton

Given a pretrained weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, 
compute its SVD:

$$
W = U \Sigma V^\top, \quad U \in \mathbb{R}^{d_\text{out} \times d_\text{out}}, 
\Sigma \in \mathbb{R}^{d_\text{out}}, V \in \mathbb{R}^{d_\text{in} \times d_\text{in}}
\tag{4.1}
$$

Keep the top-$r$:

$$
U_r = U_{:, :r}, \quad S_r = \Sigma_{:r}, \quad V_r = V_{:r, :}^\top
\tag{4.2}
$$

We store $U_r, S_r, V_r$ as **buffers** (non-trainable), typically in the 
model's dtype (FP16 or FP32).

## 4.3 Singular Value Modulation

We add a trainable scale $\gamma \in \mathbb{R}^r$, initialized to 1:

$$
\Sigma_\gamma = S_r \odot \gamma
\tag{4.3}
$$

This allows the model to amplify or suppress retained singular directions.

**Why is this useful?**
- The pretrained SVD is optimal in Frobenius norm but not for the downstream 
  task.
- $\gamma$ provides a small number of trainable parameters ($r$) that adjust 
  the spectral profile.

## 4.4 Low-Rank Residual

We add a trainable low-rank residual $AB$:

$$
A \in \mathbb{R}^{d_\text{out} \times r}, \quad 
B \in \mathbb{R}^{r \times d_\text{in}}
\tag{4.4}
$$

**Initialization:**
- $A \sim \mathcal{N}(0, 0.01^2)$ (small random)
- $B = 0$

So $AB = 0$ at initialization.

**Why a residual?** The SVD skeleton captures the top-$r$ directions. If 
these are insufficient, the residual can add capacity — but starting from 
zero avoids disrupting the pretrained state.

## 4.5 Combined Weight

The effective weight is:

$$
W_\text{SALT} = U_r \cdot \text{diag}(S_r \odot \gamma) \cdot V_r + A B
\tag{4.5}
$$

**At initialization:** $W_\text{SALT} = U_r \text{diag}(S_r) V_r$ — the 
rank-$r$ truncated SVD of $W_\text{pretrained}$.

## 4.6 Forward Pass

### 4.6.1 Naive (Our Implementation)

```python
def forward(self, x):
    adapted_S = self.S * self.gamma
    W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt  # reconstruct
    W_lora = self.A @ self.B                             # reconstruct
    W = (W_svd + W_lora).to(dtype=x.dtype)
    return F.linear(x, W, self.bias)                     # full matmul
```

**Cost.** $\mathcal{O}(d_\text{out} \cdot r \cdot d_\text{in} + d_\text{out} \cdot d_\text{in})$ 
per forward — **worse than baseline** due to reconstruction.

### 4.6.2 Fused (Future Work)

The ideal forward avoids reconstructing W:

```python
def forward(self, x):
    # 1. Project to rank space
    z = F.linear(x, self.Vt)                        # (B, L, r)
    # 2. Scale by S * gamma
    z = z * (self.S * self.gamma)
    # 3. Project back
    y_svd = F.linear(z, self.U)                     # (B, L, d_out)
    # 4. LoRA branch
    y_lora = F.linear(F.linear(x, self.B), self.A)
    return y_svd + y_lora + self.bias
```

**Cost.** $\mathcal{O}(r(d_\text{out} + d_\text{in}))$ — same as low-rank W.

**We do NOT implement the fused version.** The paper documents the naive 
version and its limitation.

## 4.7 Parameter Counting

| Component | Shape | Trainable? | Params |
|---|---|---|---|
| $U_r$ | $(d_\text{out}, r)$ | No (buffer) | $d_\text{out} \cdot r$ |
| $S_r$ | $(r,)$ | No (buffer) | $r$ |
| $V_r$ | $(r, d_\text{in})$ | No (buffer) | $r \cdot d_\text{in}$ |
| $\gamma$ | $(r,)$ | Yes | $r$ |
| $A$ | $(d_\text{out}, r)$ | Yes | $d_\text{out} \cdot r$ |
| $B$ | $(r, d_\text{in})$ | Yes | $r \cdot d_\text{in}$ |
| bias | $(d_\text{out},)$ | Yes | $d_\text{out}$ |
| **Total** | | | $2r(d_\text{out} + d_\text{in}) + 2r + d_\text{out}$ |
| **Full W** | | | $d_\text{out} \cdot d_\text{in}$ |
| **Ratio (768, r=16)** | | | **11.8×** |

## 4.8 Weight Initialization

### 4.8.1 SVD Buffers

For each projection, compute SVD of the pretrained weight and keep the 
top-$r$.

### 4.8.2 Trainable Parameters

- $\gamma \leftarrow \mathbf{1}$
- $A \sim \mathcal{N}(0, 0.01^2)$
- $B \leftarrow \mathbf{0}$
- bias $\leftarrow$ pretrained bias

## 4.9 Algorithm

**Algorithm 1:** SALT initialization
```
Input:  W ∈ R^{d_out × d_in}, bias b, rank r
Output: SALTLinear module

1: U, S, Vt ← svd(W, full_matrices=False)
2: U_r ← U[:, :r];  S_r ← S[:r];  Vt_r ← Vt[:r, :]
3: register_buffer("U", U_r)
4: register_buffer("S", S_r)
5: register_buffer("Vt", Vt_r)
6: γ ← ones(r)
7: A ← randn(d_out, r) * 0.01
8: B ← zeros(r, d_in)
9: bias ← b
10: return SALTLinear(U_r, S_r, Vt_r, γ, A, B, bias)
```

**Algorithm 2:** SALT forward (naive)
```
Input:  x ∈ R^{B × L × d_in}
Output: y ∈ R^{B × L × d_out}

1: S_γ ← S * γ
2: W_svd ← (U * S_γ) @ Vt          ▷ (d_out, d_in)
3: W_lora ← A @ B                   ▷ (d_out, d_in)
4: W ← W_svd + W_lora
5: y ← x @ W^T + bias
6: return y
```

## 4.10 Complexity Analysis

| Operation | Time | Memory |
|---|---|---|
| Naive forward | $\mathcal{O}(BL(d_\text{out} d_\text{in} + r(d_\text{out} + d_\text{in})))$ | $\mathcal{O}(d_\text{out} d_\text{in})$ |
| Fused forward | $\mathcal{O}(BL r (d_\text{out} + d_\text{in}))$ | $\mathcal{O}(r(d_\text{out} + d_\text{in}))$ |
| Baseline | $\mathcal{O}(BL d_\text{out} d_\text{in})$ | $\mathcal{O}(d_\text{out} d_\text{in})$ |

**The naive forward has extra work** from reconstruction. The fused forward 
realizes the low-rank speedup but is not implemented in this work.

## 4.11 Design Choices

### 4.11.1 Why SVD Buffers Instead of Trainable?

Buffers avoid accidental modification. If $U, S, V$ were trainable, the model 
could drift arbitrarily far from the pretrained state. Freezing them ensures 
that changes are localized to $\gamma$ and $AB$.

### 4.11.2 Why γ?

A scalar per singular value provides a minimal but expressive modulation. 
It has only $r$ parameters (16 for r=16) — negligible overhead.

### 4.11.3 Why A ~ Small Random, B = 0?

Standard LoRA init. Ensures $AB = 0$ at start, so the model matches the 
truncated SVD initialization.

### 4.11.4 Why the Same Rank for SVD and Residual?

Using a single rank parameter simplifies the API. In principle, SVD rank and 
residual rank could differ.

---