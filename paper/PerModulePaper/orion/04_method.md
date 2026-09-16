<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

ORION replaces the standard attention block with:

1. **Factorized projections** ($W_\text{up} W_\text{down}$) instead of full 
   matrices.
2. **Bifurcated heads**: kernel and full.
3. **RFF-based linear attention** for kernel heads.
4. **Softpick full attention** for full heads.

```
                Input X
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    ┌───────┐            ┌───────┐
    │  Q    │            │  K/V  │
    │ Factor│            │ Factor│
    └───┬───┘            └───┬───┘
        │                    │
        └──────┬─────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
   Kernel heads    Full heads
   (2/3)           (1/3)
       │                │
       ▼                ▼
   RFF linear      Softpick
   attention       full attention
       │                │
       └───────┬────────┘
               │
               ▼
        Concatenate + out_proj (factorized)
               │
               ▼
            Output Y
```

## 4.2 Factorized Projection

For each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

1. Compute SVD: $W = U \Sigma V^\top$.
2. Energy threshold: select smallest $r$ with $E(r) \geq \eta$.
3. Set $W_\text{up} = U_{:, :r} \Sigma_{:r, :r}^{1/2}$, $W_\text{down} = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top$.
4. Store as `up` and `down` linear layers.

**Default:** $\eta = 0.95$, min rank 4, max rank 64.

**Forward:**

$$
y = W_\text{up}(W_\text{down} x) + b
\tag{4.1}
$$

**LoRA recovery:**

$$
y = W_\text{up}(W_\text{down} x) + \frac{\alpha_L}{r_L} B(A x) + b
\tag{4.2}
$$

## 4.3 Head Bifurcation

For $H$ heads:

$$
H_f = \max(1, \lfloor H / 3 \rfloor), \quad H_k = H - H_f
\tag{4.3}
$$

For $H = 12$: $H_f = 4$, $H_k = 8$.

Kernel heads index $0, \ldots, H_k - 1$; full heads index $H_k, \ldots, H - 1$.

## 4.4 Kernel Heads (Linear Attention with RFF)

For the first $H_k$ heads:

$$
Q_\phi = \phi(Q_k), \quad K_\phi = \phi(K_k)
\tag{4.4}
$$

where $\phi(x) = [\cos(\Omega x), \sin(\Omega x)] / \sqrt{2F}$.

**Non-causal path:**

$$
KV = K_\phi^\top V, \quad Z = \sum_j K_\phi^{(j)}
\tag{4.5}
$$

$$
O_{\text{kernel}, i} = \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z}
\tag{4.6}
$$

**Causal path (chunked):**

For chunk size $C = 64$, iterate over chunks. Within each chunk, compute 
intra-chunk causal attention; across chunks, use the running state $S, Z$.

## 4.5 Full Heads (Softpick Attention)

For the last $H_f$ heads:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}}
\tag{4.7}
$$

Mask causal and padding:

$$
\text{scores}_{ij} \leftarrow \begin{cases} \text{scores}_{ij} & j \leq i \text{ and valid} \\ -10^4 & \text{otherwise} \end{cases}
\tag{4.8}
$$

Softpick:

$$
\text{attn}_{ij} = \frac{\text{ReLU}(e^{\text{scores}_{ij}} - 1)}{\sum_k \text{ReLU}(e^{\text{scores}_{ik}} - 1) + \epsilon}
\tag{4.9}
$$

**Note on masked positions:** For masked positions with score $-10^4$, 
$e^{-10^4} \approx 0$, $\text{ReLU}(-1) = 0$. So masked keys contribute 
zero. **But** if all keys in a row are masked, the denominator becomes 0 
(plus epsilon) and the output is all zeros — no attention.

## 4.6 Output Fusion

Concatenate kernel and full heads:

$$
O = \text{concat}(O_{\text{kernel}}, O_{\text{full}})
\tag{4.10}
$$

Then apply the factorized output projection.

## 4.7 Parameter Analysis

For a single projection $W \in \mathbb{R}^{d \times d}$:

| Variant | Params |
|---|---|
| Original | $d^2$ |
| Factorized (rank $r$) | $r(d + d) = 2rd$ |
| **Ratio** | $\frac{d}{2r}$ |

For $d = 768$, $r = 64$: **6× reduction**.

For Q projection, ORION uses $r = 64$ and $d = 768$:

$$
768 \times 768 = 589,824 \quad \rightarrow \quad 2 \times 64 \times 768 = 98,304
$$

**Per layer (DistilBERT):**
- Q, K, V, O factorized: $4 \times 98,304 = 393,216$ vs original $4 \times 589,824 = 2,359,296$.
- **Savings per layer: ~1.97M params.**

For 6 layers: **~11.8M savings.**

## 4.8 Complexity Analysis

| Component | Time |
|---|---|
| Kernel heads (linear) | $\mathcal{O}(L F d)$ |
| Full heads (softpick) | $\mathcal{O}(L^2 d)$ |
| Fusion | $\mathcal{O}(L d)$ |

**Total:** $\mathcal{O}(L^2 d) \cdot (H_f / H) + \mathcal{O}(L F d) \cdot (H_k / H)$.

For $H_f / H = 1/3$: **attention cost reduced by ~3×** compared to full 
attention (assuming linear cost is negligible for short sequences).

## 4.9 Algorithm

**Algorithm 1:** ORION forward
```
Input:  X ∈ R^{B×L×d}
Output: Y ∈ R^{B×L×d}

1: # Factorized projections
2: Q_all ← W_Q_up(W_Q_down(X))
3: K_all ← W_K_up(W_K_down(X))
4: V_all ← W_V_up(W_V_down(X))
5:
6: if RoPE: (Q, K) ← Rotary(Q, K)
7: if GQA: (K, V) ← repeat_kv(K, V)
8:
9: # Split heads
10: Q_kernel, Q_full ← Q[:, :H_k], Q[:, H_k:]
11: K_kernel, K_full ← K[:, :H_k], K[:, H_k:]
12: V_kernel, V_full ← V[:, :H_k], V[:, H_k:]
13:
14: # Kernel heads
15: O_kernel ← RFF_linear_attention(Q_kernel, K_kernel, V_kernel)
16:
17: # Full heads
18: O_full ← softpick_attention(Q_full, K_full, V_full)
19:
20: # Concatenate + output
21: O ← concat(O_kernel, O_full)
22: Y ← W_O_up(W_O_down(O))
23: return Y
```

## 4.10 The `copy_weights` Bug

**Critical.** The `copy_weights` method contains:

```python
missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
if missing:
    try:
        from framework.core.adapters import UniversalAdapter
        adapter = UniversalAdapter()
        for k in missing:
            w, b = adapter.get_weight_bias(src, k)
            if w is not None:
                extracted[k] = (w.clone(), b.clone() if b is not None else None)
    except Exception as e:
        print(f'copy_weights failed: {e}')
        raise RuntimeError(f"copy_weights failed: {e}")  # ← BUG
```

**Two failure modes:**

1. **UniversalAdapter raises:** → RuntimeError propagates.
2. **UniversalAdapter returns None for all missing:** → `extracted[k]` stays None, then `missing_after` check triggers another RuntimeError.

In both cases, the injection aborts. If the caller catches the exception, 
the module keeps **random init weights** → collapse.

**Fix:**

```python
except Exception as e:
    logger.debug(f"UniversalAdapter failed: {e}")
    # Continue
```

## 4.11 Theoretical Notes

### 4.11.1 SVD Optimality

The truncated SVD is the optimal rank-$r$ approximation in Frobenius norm 
(Eckart-Young).

### 4.11.2 RFF Approximation

RFF approximates the Gaussian kernel $\exp(-\|q - k\|^2 / 2)$. The 
approximation error decreases as $\mathcal{O}(1/\sqrt{F})$.

### 4.11.3 Softpick Properties

ORION's softpick produces **sparse** attention: for $x_i$ sufficiently 
negative, $\text{ReLU}(e^{x_i} - 1) = 0$. This differs from softmax which 
always produces nonzero weights.

**Potential issue:** If all $x_i < 0$, all weights are zero → output is 
zero. This happens on masked rows.

---