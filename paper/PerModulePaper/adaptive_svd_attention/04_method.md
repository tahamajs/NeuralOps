# 4. Method

## 4.1 Overview

Adaptive SVD Attention replaces each of the four attention projections 
$\{W_Q, W_K, W_V, W_O\}$ with:

$$
W \approx U_r V_r + B A
\tag{4.1}
$$

where:

- $r$ is **adaptively selected per projection** via energy thresholding.
- $U_r, V_r$ are frozen SVD factors.
- $B A$ is a trainable residual adapter with $B = 0$ initially.

**Figure 4.1** shows the pipeline.

```
    Weight W ∈ R^{d×d}
              │
              ▼
    ┌───────────────────┐
    │  SVD              │
    │  U Σ V^T          │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Energy-based     │
    │  rank selection   │
    │  r* : E(r*) ≥ τ   │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Symmetric split  │
    │  U_r = U √Σ       │
    │  V_r = √Σ V^T     │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │  Freeze U_r, V_r  │
    │  Add LoRA BA      │
    └───────────────────┘
```

## 4.2 Energy-Thresholded Rank Selection

For each projection $W$:

**Step 1:** Compute SVD: $W = U \Sigma V^\top$.

**Step 2:** Compute cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{j=1}^{d} \sigma_j^2}
\tag{4.2}
$$

**Step 3:** Find minimal rank:

$$
r^* = \min\{r : E(r) \geq \tau\}
\tag{4.3}
$$

**Step 4:** Clamp to $[\min\_rank, \max\_rank]$:

$$
r^* \leftarrow \max(\min(r^*, \text{max\_rank}), \text{min\_rank})
\tag{4.4}
$$

We use $\tau = 0.92$, $\min\_rank = 16$, $\max\_rank = d$.

### 4.2.1 Why Energy Thresholding?

Energy thresholding is **principled** — it directly bounds the relative 
Frobenius error:

$$
\frac{\|W - W_{r^*}\|_F}{\|W\|_F} \leq \sqrt{1 - \tau}
\tag{4.5}
$$

For $\tau = 0.92$: error ≤ 28.3% of the Frobenius norm. This is a **hard 
guarantee**, unlike a fixed rank which gives no such bound.

### 4.2.2 Why Per-Projection?

Different projections have different spectral decay:

- **Attention Q, K:** often have rapidly decaying spectra (low intrinsic rank).
- **Attention V, O:** sometimes flatter (higher intrinsic rank).

A fixed rank either over-compresses V and O or under-compresses Q and K. 
Adaptive rank allocates capacity where it's needed.

## 4.3 Symmetric $\sqrt{\Sigma}$ Split

Given the selected rank $r^*$, we compute:

$$
U_r = U_{:, :r^*} \Sigma_{:r^*, :r^*}^{1/2}
\tag{4.6}
$$

$$
V_r = \Sigma_{:r^*, :r^*}^{1/2} V_{:r^*, :}^\top
\tag{4.7}
$$

Then $W \approx U_r V_r$.

**Why symmetric?** Both factors have singular values bounded by $\sqrt{\sigma_1}$, 
which keeps FP16 underflow risk low for both.

## 4.4 Residual Adapter

We freeze $U_r, V_r$ and add a trainable residual:

$$
\Delta W = B A, \quad A \in \mathbb{R}^{r_a \times d}, \quad B \in \mathbb{R}^{d \times r_a}
\tag{4.8}
$$

with:

- $A \sim \text{Kaiming-Uniform}$
- $B = 0$ (so $\Delta W = 0$ at init)
- $r_a = 16$ (residual rank)

The full forward pass:

$$
y = U_r V_r x + B A x + b
\tag{4.9}
$$

**Parameter count:**

| Component | Params |
|---|---|
| $U_r$ | $d \cdot r^*$ |
| $V_r$ | $r^* \cdot d$ |
| $A$ | $r_a \cdot d$ |
| $B$ | $d \cdot r_a$ |
| **Total** | $2 d r^* + 2 d r_a$ |

vs baseline $d^2$. Ratio:

$$
\frac{d^2}{2 d (r^* + r_a)} = \frac{d}{2(r^* + r_a)}
\tag{4.10}
$$

## 4.5 All Four Projections

We apply the same procedure to $W_Q, W_K, W_V, W_O$.

**Parameter reduction (per block):**

| Projection | Baseline | Factorized (r=128, r_a=16) |
|---|---|---|
| $W_Q$ | $d^2$ | $2d \cdot 144$ |
| $W_K$ | $d^2$ | $2d \cdot 144$ |
| $W_V$ | $d^2$ | $2d \cdot 144$ |
| $W_O$ | $d^2$ | $2d \cdot 144$ |
| **Total** | $4 d^2$ | $8 d \cdot 144$ |

For $d = 768$:

| Component | Baseline | Ours |
|---|---|---|
| Attention params | 2,359,296 | 884,736 |
| **Ratio** | | **2.67×** |

## 4.6 T5 Relative Bias Preservation

T5's relative attention bias is stored as an embedding `relative_attention_bias` 
of shape $(num\_buckets, num\_heads)$. We copy it to the new module:

```python
if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
    self.relative_attention_bias = src.relative_attention_bias
    self.has_relative_attention_bias = True
```

The bias is added to attention scores after the Q, K projections (which are 
factorized).

## 4.7 Cross-Attention Support

For encoder-decoder attention:

```python
key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
is_cross_attention = key_value_states is not None
kv_input = key_value_states if is_cross_attention else hidden_states
```

K, V are projected from `kv_input`; Q from `hidden_states`. Factorized 
projections work uniformly.

## 4.8 Algorithm

**Algorithm 1:** Adaptive SVD Attention forward pass
```
Input:  X, attention_mask, position_embeddings (optional)
        encoder_hidden_states (optional, for cross-attention)
Output: Y

 1: if cross_attention: kv_input ← encoder_hidden_states
 2: else: kv_input ← X
 3: Q ← query(X)                     ▷ U_r V_r + B A
 4: K ← key(kv_input)
 5: V ← value(kv_input)
 6: Q, K, V ← reshape to multi-head
 7: if RoPE: Q, K ← rotary(Q, K)
 8: if GQA:  K, V ← repeat_kv(K, V)
 9: scores ← Q K^T / √d_h
10: if relative_bias: scores += bias
11: if causal: scores += causal_mask
12: scores = mask(scores, attention_mask)
13: A = softmax(scores)
14: context ← A V
15: context ← reshape
16: if nested: return context
17: out ← out_proj(context)
18: return out
```

## 4.9 Complexity Analysis

For a single attention block:

| Component | Time | Params |
|---|---|---|
| Q proj | $\mathcal{O}(L d r^*)$ | $2 d r^* + 2 d r_a$ |
| K proj | $\mathcal{O}(L d r^*)$ | $2 d r^* + 2 d r_a$ |
| V proj | $\mathcal{O}(L d r^*)$ | $2 d r^* + 2 d r_a$ |
| O proj | $\mathcal{O}(L d r^*)$ | $2 d r^* + 2 d r_a$ |
| Attention | $\mathcal{O}(L^2 d)$ | 0 |

**Projection FLOPs reduction:** $d / r^*$.

## 4.10 Rank Selection Statistics

For a typical trained model, the selected ranks are:

| Projection | Mean $r^*$ | Min | Max |
|---|---|---|---|
| Q | 128 | 96 | 160 |
| K | 128 | 96 | 160 |
| V | 192 | 128 | 224 |
| O | 192 | 128 | 224 |

Across all layers, average rank: **160**.

---