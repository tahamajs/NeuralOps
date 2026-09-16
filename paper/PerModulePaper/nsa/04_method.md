<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

NSA computes a hybrid sparse attention mask that combines:
1. A **fixed sliding window**.
2. A **dynamic top-$k$** selection of distant keys.

**Figure 4.1** illustrates the pipeline.

```
      Q, K, V ∈ R^{B × H × L × d_h}
                 │
                 ▼
      ┌────────────────────────┐
      │  S = Q K^T / √d_h      │
      └──────────┬─────────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │  Local window mask     │
      │  W_ij = |i-j| <= w     │
      └──────────┬─────────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │  Top-k global          │
      │  (outside window)      │
      └──────────┬─────────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │  Union + causal + pad  │
      └──────────┬─────────────┘
                 │
                 ▼
      ┌────────────────────────┐
      │  Softmax on S + mask   │
      └──────────┬─────────────┘
                 │
                 ▼
              O = P · V
```

## 4.2 Local Window Mask

For window size $w$:

$$
\mathcal{W}_{ij} = \begin{cases} 1 & |i - j| \leq w \\ 0 & \text{otherwise} \end{cases}
\tag{4.1}
$$

**Causal version:**

$$
\mathcal{W}_{ij} = \mathbb{1}[j \leq i \text{ and } i - j \leq w]
\tag{4.2}
$$

## 4.3 Dynamic Top-K Global

### 4.3.1 Score Computation

Given scores $s_{ij} = Q_i^\top K_j / \sqrt{d_h}$, we mask out the local 
window and padding:

$$
\tilde{s}_{ij} = \begin{cases} s_{ij} & j \notin \mathcal{W}_i \text{ and valid}_j \\ -\infty & \text{otherwise} \end{cases}
\tag{4.3}
$$

**Causal-aware:** For decoders, also mask future positions:

$$
\tilde{s}_{ij} = -\infty \quad \text{if } j > i
\tag{4.4}
$$

### 4.3.2 Top-K Selection

We select the top-$k$ positions:

$$
\text{topk-ind}_i = \arg\text{topk}_{k}(\tilde{s}_i)
\tag{4.5}
$$

Implemented via `torch.topk`.

### 4.3.3 Why Top-K on Masked Scores?

We could instead select top-$k$ over all scores and then remove window 
positions. But that would waste $k$ budget on window entries. By masking 
window first, we ensure all $k$ selections are **outside** the window, 
maximizing the union coverage.

## 4.4 Union Mask

The combined mask is:

$$
\mathcal{S}_{ij} = \mathcal{W}_{ij} \lor \text{topk-mask}_{ij}
\tag{4.6}
$$

**Final mask (with padding and causality):**

$$
\mathcal{S}_{ij}^\text{final} = \mathcal{S}_{ij} \land \text{valid}_j \land \text{causal}_{ij}
\tag{4.7}
$$

## 4.5 Short Sequence Handling

When $L \leq 2w + 1$, every key is already in the window. The top-$k$ step 
is skipped:

$$
\mathcal{S}_{ij}^\text{final} = \mathcal{W}_{ij} \land \text{valid}_j \land \text{causal}_{ij}
\tag{4.8}
$$

This avoids unnecessary computation for short sequences.

## 4.6 Efficient Top-K Computation

**Full computation** (naive):

```python
# Mask local window and padding
masked_scores = scores.masked_fill(local_mask_4d | ~key_valid_4d, -1e4)
# Top-k
_, topk_idx = torch.topk(masked_scores, k=k, dim=-1)
# Build mask
topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
```

**Complexity.** `torch.topk` on an $L \times L$ matrix costs 
$\mathcal{O}(L^2)$ time. This is not better than dense attention, but:

- We only care about **memory** (masked softmax), not time.
- For $L > 512$, the sparse mask saves memory in subsequent operations.
- A custom top-$k$ kernel could reduce time, but that's future work.

## 4.7 Causality Enforcement

**Critical ordering.** We apply causal masking **before** top-$k$ selection 
to prevent future leakage:

```python
# 1. Mask local window + padding + causal
non_local_scores = scores.masked_fill(local_mask | ~key_valid_4d, -1e4)
if is_dec:
    non_local_scores = non_local_scores.masked_fill(~causal_tril, -1e4)

# 2. Top-k on non-local + non-future
_, topk_ind = torch.topk(non_local_scores, k=k, dim=-1)

# 3. Build mask
topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_ind, True)
```

**Verify.** For decoder, $j > i$ positions are never selected.

## 4.8 Complete Forward Pass

**Algorithm 1:** NSA forward
```
Input:  Q, K, V ∈ R^{B × H × L × d_h}
        attention_mask, is_causal, w, k
Output: context ∈ R^{B × H × L × d_h}

 1: S ← Q K^T / sqrt(d_h)                    # (B, H, L, L)
 2: valid ← get_token_valid_mask(attention_mask)
 3:
 4: # Local window mask
 5: idx ← arange(L)
 6: if is_causal:
 7:     local_mask ← (idx_q - idx_k <= w) & (idx_k <= idx_q)
 8: else:
 9:     local_mask ← |idx_q - idx_k| <= w
10:
11: # Union mask
12: if L <= 2w + 1:
13:     total_mask ← local_mask & valid
14: else:
15:     # Mask out local window + padding
16:     non_local ← S.masked_fill(local_mask | ~valid, -1e4)
17:     if is_causal:
18:         non_local ← non_local.masked_fill(~causal_tril, -1e4)
19:     # Top-k selection
20:     _, topk_ind ← topk(non_local, k=min(k, L - (2w + 1)))
21:     topk_mask ← zeros_like(S).scatter(-1, topk_ind, True)
22:     total_mask ← (local_mask | topk_mask) & valid
23:
24: # Final causal (redundant but safe)
25: if is_causal:
26:     total_mask ← total_mask & causal_tril
27:
28: # Softmax with mask
29: S_masked ← S.masked_fill(~total_mask, -1e4)
30: P ← softmax(S_masked)
31: P ← dropout(P)
32: O ← P V
33: return O
```

## 4.9 Complexity Analysis

| Component | Dense | NSA |
|---|---|---|
| Score matrix | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ |
| Top-$k$ | — | $\mathcal{O}(B H L \cdot k)$ |
| Mask construction | — | $\mathcal{O}(B H L^2)$ |
| Softmax | $\mathcal{O}(B H L^2)$ | $\mathcal{O}(B H L^2)$ (masked) |
| Attention output | $\mathcal{O}(B H L^2 d_h)$ | $\mathcal{O}(B H L (2w+k) d_h)$ |

**Note.** NSA does **not** reduce theoretical complexity (top-$k$ is still 
$\mathcal{O}(L^2)$). The savings are in the **mask density**:

$$
\rho = \frac{2w + 1 + k}{L}
\tag{4.9}
$$

For $w = 32$, $k = 16$, $L = 128$: $\rho = 0.63$.
For $L = 2048$: $\rho = 0.04$.

**To realize speedup**, one needs sparse matmul kernels that skip the 
$-10^4$-masked entries. This is future work.

## 4.10 Parameter Analysis

| Component | Baseline | NSA |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | Standard | Same |
| **Extra parameters** | 0 | **0** |
| LoRA | 0 | $2 r (\cdots)$ |

**NSA adds zero parameters.** The window size $w$ and top-$k$ count are 
hyperparameters, not trainable.

## 4.11 Theoretical Notes

### 4.11.1 Attention Mass Preservation

**Claim.** For any $\epsilon > 0$, there exists a window $w$ and top-$k$ 
count $k$ such that the preserved attention mass is at least $1 - \epsilon$:

$$
\sum_{j \in \mathcal{S}_i} P_{ij} \geq 1 - \epsilon
\tag{4.10}
$$

**Proof sketch.** Set $w$ large enough to cover local mass, and $k$ large 
enough to cover the tail. $\square$

### 4.11.2 Top-K Selection Quality

**Claim.** Top-$k$ on masked scores $\tilde{s}_{ij}$ yields the same top-$k$ 
set as top-$k$ on $s_{ij}$ outside the window.

**Proof.** Masking with $-\infty$ preserves the ordering of valid entries. 
$\square$

### 4.11.3 Causality Guarantee

**Claim.** For decoders, NSA never attends to future positions.

**Proof.** Two mechanisms:
1. Local mask: $i - j \leq w$ AND $j \leq i$.
2. Top-$k$ mask: computed on scores that have been causal-masked.

Combined with a final causal AND, causality is guaranteed. $\square$

### 4.11.4 Coverage Bound

**Claim.** The number of attended keys is bounded by:

$$
|\mathcal{S}_i| \leq 2w + 1 + k
\tag{4.11}
$$

**Proof.** The local window has $2w + 1$ keys; the top-$k$ adds at most $k$ 
more. $\square$

---