<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

RALA augments standard linear attention with three components:

```
    Input X ∈ R^{B × L × d}
            │
            ▼
    ┌─────────────────────────────┐
    │  Q/K/V projections (LoRA)   │
    └──────────────┬──────────────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
    ┌──────────────┐  ┌──────────────┐
    │  φ = softplus│  │  φ = softplus│
    │  Q' = φ(Q)   │  │  K' = φ(K)   │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           │        ┌────────┴────────┐
           │        ▼                 ▼
           │   ┌─────────┐      ┌─────────────┐
           │   │ KV = K'ᵀV│     │ z = K'ᵀ·1  │
           │   └────┬────┘      └──────┬──────┘
           │        │                  │
           └────────┴──────────┬───────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  lin_out = Q'·KV / Q'·z │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │ rank_mat = U Vᵀ     │
                    │ aug_out = Q'·rank_mat/z │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ out = lin_out + α·aug_out │
                    └──────────────────────┘
```

**Key idea.** The augmented branch contributes a **rank-$r$ correction** 
to the linear attention output, controlled by a learned scalar $\alpha$.

## 4.2 Softplus Kernel

We define the kernel feature map as:

$$
\phi(x) = \mathrm{softplus}(x) = \log(1 + e^x)
\tag{4.1}
$$

**Numerical stability.** For large positive $x$, $\log(1 + e^x) \approx x$; 
for large negative $x$, $\log(1 + e^x) \approx e^x \approx 0$. PyTorch's 
`F.softplus` uses the numerically stable form:

$$
\mathrm{softplus}(x) = \max(x, 0) + \log(1 + e^{-|x|})
\tag{4.2}
$$

**Why not elu+1?** The function $\mathrm{elu}(x) + 1$ has a discontinuity 
in its second derivative at $x = 0$. Softplus is $C^\infty$.

## 4.3 Rank-Augmented Linear Attention

We parameterize per-head matrices $U_h, V_h \in \mathbb{R}^{d \times r}$ and 
form:

$$
M_h = U_h V_h^\top \in \mathbb{R}^{d \times d}
\tag{4.3}
$$

The augmented output is:

$$
\text{aug}_i^{(h)} = \frac{\phi(q_i^{(h)})^\top M_h}{\phi(q_i^{(h)})^\top z_i^{(h)}}
\tag{4.4}
$$

Note that $M_h$ has rank at most $r$. The augmented branch is **not causal** 
by itself — it applies a static (data-independent) matrix to the query. It 
does not violate causality because $M_h$ is a fixed linear map; no 
information from position $j > i$ is used.

## 4.4 The Scalar Gate

The final output combines linear and augmented outputs:

$$
y_i^{(h)} = \text{lin}_i^{(h)} + \alpha \cdot \text{aug}_i^{(h)}
\tag{4.5}
$$

where $\alpha$ is a single learned scalar **shared across all heads and 
layers** (in this implementation).

**Initialization.** $\alpha = 0$ at start, so the model reduces to pure 
linear attention. The rank augmentation is learned on top.

**Gradient flow.** Since $\alpha$ is scalar, it aggregates gradient from all 
heads and positions. This may be limiting — a per-head gate might be better. 
We note this as future work.

## 4.5 Causal Prefix Scan

For decoder models, we compute prefix sums:

$$
S_i = \sum_{j \le i} \phi(k_j) v_j^\top, \quad
z_i = \sum_{j \le i} \phi(k_j)
\tag{4.6}
$$

The output uses the prefix sums at position $i$:

$$
y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
\tag{4.7}
$$

**Our implementation.** We use a Python loop over sequence length:

```python
for i in range(seq_len):
    S = S + einsum("bhd,bhe->bhde", k_i, v_i)
    z = z + k_i
    num_i = einsum("bhd,bhde->bhe", q_i, S)
    denom_i = einsum("bhd,bhd->bh", q_i, z).unsqueeze(-1)
    num_list.append(num_i)
    denom_list.append(denom_i)
```

**Cost.** $\mathcal{O}(L)$ sequential steps, each with $\mathcal{O}(BH d^2)$ 
work. For $L = 128$, this is 128 kernel launches per head, which dominates 
runtime.

**Future work.** Replace with a parallel associative scan. This would reduce 
to $\mathcal{O}(\log L)$ depth and dramatically improve throughput.

## 4.6 The Full Forward Pass

**Algorithm 1:** RALA forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V    ▷ LoRA projections
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA:  (K, V) ← repeat_kv(K, V)
 4:
 5: Q' ← softplus(Q);  K' ← softplus(K)   ▷ Kernel
 6: K' ← K' ⊙ mask;    V ← V ⊙ mask       ▷ Padding
 7:
 8: if causal:
 9:     for i = 1 to L:
10:         S ← S + K'_i V_i^T
11:         z ← z + K'_i
12:         lin_i ← Q'_i S / (Q'_i z)
13:         aug_i ← Q'_i M / (Q'_i z)
14: else:
15:     KV ← K'^T V;  z ← K'^T 1
16:     lin ← Q' KV / (Q' z)
17:     aug ← Q' M / (Q' z)
18:
19: out ← lin + α · aug
20: if not is_nested_encoder: out ← out_proj(out)
21: return out
```

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V projection | $\mathcal{O}(BL d^2)$ | $\mathcal{O}(BLd)$ |
| Kernel apply | $\mathcal{O}(BLd)$ | $\mathcal{O}(BLd)$ |
| Prefix scan (causal) | $\mathcal{O}(L)$ sequential | $\mathcal{O}(BH d^2)$ |
| Non-causal | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BHd^2)$ |
| Augmentation | $\mathcal{O}(BL d r)$ | $\mathcal{O}(BH d r)$ |
| Output projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |

**Causal path is the bottleneck.** The Python loop adds O(L) kernel launches.

## 4.8 Parameter Analysis

RALA adds:

| Component | Per layer | Total (DistilBERT) |
|---|---|---|
| $U$ | $H \cdot d \cdot r$ | 6 layers × 6,144 = 36,864 |
| $V$ | $H \cdot d \cdot r$ | 36,864 |
| $\alpha$ | 1 | 6 |
| LoRA ($4$ proj × $r_L$) | $4 \cdot 2 r_L d$ | 6 layers × 98,304 = 589,824 |
| **Total added** | | **~663,558** |
| **Percentage** | | **+0.99%** |

**RALA increases parameter count.** It does not compress the model.

## 4.9 Design Choices

### 4.9.1 Why Softplus?

| Kernel | Smooth? | Positive? | Unbounded? |
|---|---|---|---|
| elu+1 | ❌ (kink at 0) | ✅ | ❌ (bounded by 1 for negative) |
| ReLU | ❌ (kink at 0) | ✅ (weakly) | ✅ |
| Softplus | ✅ | ✅ | ✅ |
| $e^x$ | ✅ | ✅ | ✅ but unstable |

Softplus is the unique choice satisfying all three properties.

### 4.9.2 Why Rank-$r$ per Head?

A single shared matrix $M$ across heads would waste capacity: different heads 
learn different interaction patterns. Per-head factorization gives each head 
its own correction at $\mathcal{O}(r d)$ parameters per head.

### 4.9.3 Why Scalar Gate?

A scalar gate has minimal overhead and is interpretable: $\alpha = 0$ means 
pure linear attention; $\alpha = 1$ means full augmentation. A per-head gate 
would be more expressive but adds $H$ parameters per layer.

### 4.9.4 Why Frozen $U, V$?

Our implementation freezes $U, V$ when `freeze_base = True`. This is 
unusual: normally we would train them. The rationale is that `U, V` are 
inside the "base" attention and only the LoRA adapters are meant to be 
trained. **However, this means the augmentation is a fixed random projection 
scaled by $\alpha$ — not a learned one.** This is a real limitation and we 
document it in §10.

## 4.10 Initialization

```python
self.U = nn.Parameter(torch.randn(H, d, r) * 0.01)
self.V = nn.Parameter(torch.randn(H, d, r) * 0.01)
self.alpha = nn.Parameter(torch.tensor(0.0))
```

$U, V$ use small-variance random init; $\alpha = 0$ disables the 
augmentation at start.

---