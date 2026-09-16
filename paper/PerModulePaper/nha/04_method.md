<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

NHA runs two attention streams in parallel and combines them via a token-level 
gate:

1. **Local stream** $O^\text{local}$: sliding-window attention.
2. **Global stream** $O^\text{global}$: RFF linear attention.
3. **Gate** $g$: per-token sigmoid.
4. **Fusion** $O = g \odot O^\text{local} + (1 - g) \odot O^\text{global}$.

**Figure 4.1** illustrates the pipeline.

```
        Q, K, V ∈ R^{B × H × L × d_h}
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌──────────┐      ┌──────────┐
   │  Local   │      │  Global  │
   │  window  │      │   RFF    │
   │ O_local  │      │ O_global │
   └────┬─────┘      └────┬─────┘
        │                 │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │  Gate g(x)      │
        │  g·O_l+(1-g)·O_g│
        └────────┬────────┘
                 ▼
              out_proj
```

## 4.2 Local Window Stream

### 4.2.1 Definition

For window size $w$:

$$
S_{ij} = \frac{Q_i^\top K_j}{\sqrt{d_h}}, \quad
\text{mask}_{ij} = \mathbb{1}[|i - j| \leq w]
\tag{4.1}
$$

$$
O_i^\text{local} = \text{softmax}(S \odot \text{mask})_i \cdot V
\tag{4.2}
$$

### 4.2.2 Chunked Computation

To bound memory, we process the query dimension in chunks of 64:

```python
for i in range(0, N, chunk):
    end = min(i + chunk, N)
    q_block = Q[:, :, i:end]
    k_start = max(0, i - window)
    k_end = min(N, end + window)
    k_block = K[:, :, k_start:k_end]
    # ... compute scores, mask, softmax ...
```

This gives $\mathcal{O}(B H C (2w + C))$ peak memory instead of 
$\mathcal{O}(B H L^2)$.

### 4.2.3 Causal Masking

For decoders:

$$
\text{mask}_{ij}^\text{causal} = \mathbb{1}[j \leq i \text{ and } i - j \leq w]
\tag{4.3}
$$

## 4.3 Global RFF Stream

### 4.3.1 RFF Projection

We build $\Omega \in \mathbb{R}^{d_h \times m}$ via block QR:

```python
blocks = []
remaining = m
while remaining > 0:
    block_size = min(d_h, remaining)
    g = torch.randn(d_h, block_size)
    q, _ = torch.linalg.qr(g)
    blocks.append(q)
    remaining -= block_size
W = torch.cat(blocks, dim=1) * math.sqrt(d_h)
```

**Orthogonality.** Each block $Q_i$ has orthonormal columns. The scale 
$\sqrt{d_h}$ matches the expected variance of Gaussian projections.

### 4.3.2 Feature Map

$$
\phi(x) = \frac{1}{\sqrt{m}} \left(\mathrm{elu}(\Omega^\top x) + 1\right)
\tag{4.4}
$$

**Properties:**
- **Non-negative:** $\mathrm{elu}(x) > -1$ → $\phi > 0$.
- **Bounded:** $\phi \in [0, \infty)$.
- **Smooth:** ELU is $C^1$.

### 4.3.3 Non-Causal Computation

For encoders:

$$
KV = \phi(K)^\top V \in \mathbb{R}^{m \times d_h}, \quad
Z = \sum_j \phi(K_j) \in \mathbb{R}^m
\tag{4.5}
$$

$$
O_i^\text{global} = \frac{\phi(Q_i)^\top KV}{\phi(Q_i)^\top Z + \epsilon}
\tag{4.6}
$$

### 4.3.4 Causal Prefix Scan (Chunked)

For decoders, we use a chunked prefix scan with cumulative state:

$$
S \in \mathbb{R}^{B \times H \times m \times d_h}, \quad
Z \in \mathbb{R}^{B \times H \times m}
$$

Initialize $S = 0$, $Z = 0$. For each chunk $[start, end)$:

1. Compute intra-chunk causal scores:
   $$
   A^\text{intra} = \phi(Q_c) \phi(K_c)^\top \odot \text{tril}(1)
   \tag{4.7}
   $$

2. Compute inter-chunk contribution:
   $$
   O^\text{inter} = \phi(Q_c) S
   \tag{4.8}
   $$

3. Combine:
   $$
   O_c = \frac{A^\text{intra} V_c + O^\text{inter}}{(A^\text{intra} \cdot 1 + \phi(Q_c) Z) + \epsilon}
   \tag{4.9}
   $$

4. Update state:
   $$
   S \leftarrow S + \phi(K_c)^\top V_c, \quad Z \leftarrow Z + \sum \phi(K_c)
   \tag{4.10}
   $$

**Complexity.** $\mathcal{O}(B H L m d_h)$ time, $\mathcal{O}(B H m d_h)$ memory.

### 4.3.5 Numerical Stability

We compute the RFF stream in **FP32** inside an autocast-disabled block:

```python
with torch.autocast(device_type=..., enabled=False):
    phi_Q_f = phi_Q.float()
    phi_K_f = phi_K.float()
    V_f = V_masked.float()
    # ... computation ...
```

This avoids FP16 overflow for $L > 512$.

## 4.4 Token-Level Gate

### 4.4.1 Parameterization

$$
g_i = \sigma(w^\top x_i + b)
\tag{4.11}
$$

where $x_i$ is the input hidden state at position $i$, and $w \in \mathbb{R}^d$, 
$b \in \mathbb{R}$.

### 4.4.2 Initialization

We initialize $w = 0$ and $b = 3.0$, so:

$$
g_i = \sigma(0 + 3) = \sigma(3) \approx 0.95 \quad \text{for all } i
\tag{4.12}
$$

**Why bias toward local?** The local window matches the pretrained model's 
short-range behavior. Starting global-leaning would immediately degrade 
accuracy.

### 4.4.3 Per-Token vs Per-Head

NHA uses **per-token** gating (single scalar per position), while HALO-ReGLA 
uses per-head. Per-token is:

- **Parameter-efficient:** $d + 1$ parameters instead of $d H + H$.
- **Position-aware:** Can adapt to local vs global needs per position.
- **Simpler:** No head dimension to manage.

Empirically, per-token works as well as per-head for our tasks.

## 4.5 Complete Forward Pass

**Algorithm 1:** NHA forward
```
Input:  hidden_states ∈ R^{B × L × d}
        attention_mask, position_embeddings, is_causal
        window_size w, rff_features m
Output: output ∈ R^{B × L × d}

 1: Q, K, V ← proj(hidden_states)
 2: Q, K ← apply_rope(Q, K, position_embeddings)
 3: K, V ← repeat_kv(K, V, H / H_kv) if GQA
 4:
 5: # Local stream (chunked)
 6: O_local ← sliding_window_attention(Q, K, V, w, is_causal)
 7:
 8: # Global RFF stream
 9: φ_Q ← elu(Ω^T Q) + 1
10: φ_K ← elu(Ω^T K) + 1
11: if is_causal:
12:     (S, Z) ← initialize_cumulative_state()
13:     for chunk in chunks(L, 64):
14:         O_global[chunk] ← RFF_chunk(φ_Q, φ_K, V, S, Z)
15:         (S, Z) ← update_state(S, Z, φ_K[chunk], V[chunk])
16: else:
17:     KV ← φ_K^T V
18:     Z ← sum(φ_K)
19:     O_global ← φ_Q @ KV / (φ_Q @ Z + ε)
20:
21: # Token-level gate
22: g ← sigmoid(gate(hidden_states))         # (B, L, 1)
23: O ← g * O_local + (1 - g) * O_global
24: O ← O * valid_mask
25: O ← reshape and out_proj
26: return O
```

## 4.6 Complexity Analysis

| Component | Local | Global RFF | NHA (total) |
|---|---|---|---|
| Time | $\mathcal{O}(L w d_h)$ | $\mathcal{O}(L m d_h)$ | $\mathcal{O}(L (w + m) d_h)$ |
| Memory | $\mathcal{O}(L w)$ | $\mathcal{O}(m d_h)$ | $\mathcal{O}(L w + m d_h)$ |
| vs Dense | $\mathcal{O}(L^2 d_h)$ | — | **$\mathcal{O}(L (w + m) d_h)$** |

**Key insight.** NHA is **linear** in $L$ for both branches. This is the 
primary efficiency benefit.

## 4.7 Parameter Analysis

| Component | Baseline | NHA |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Gate** | 0 | $d + 1$ |
| **RFF proj** | 0 | $d_h m$ (buffer, not trainable) |
| LoRA | 0 | $2 r (\cdots)$ |

**Total: same as MHA + $d + 1$ parameters (gate) + RFF buffer.**

## 4.8 Theoretical Notes

### 4.8.1 Kernel Approximation

With orthogonal $\Omega$ and $m \to \infty$:

$$
\phi(q)^\top \phi(k) \xrightarrow{m \to \infty} \mathbb{E}_\omega[\ldots] = k_\text{exp}(q, k)
\tag{4.13}
$$

where $k_\text{exp}$ is the exp kernel.

### 4.8.2 Gate Convergence

The gate converges to a stationary point where:

$$
\frac{\partial \mathcal{L}}{\partial g_i} = (\nabla_O \mathcal{L}) \cdot (O^\text{local}_i - O^\text{global}_i) = 0
\tag{4.14}
$$

or hits the boundary ($g_i \to 0$ or $g_i \to 1$).

### 4.8.3 Causality

For causal models, the chunked prefix scan ensures:

$$
O_i^\text{global} = f(\phi(Q_j), \phi(K_j), V_j : j \leq i)
\tag{4.15}
$$

No future information leaks.

---