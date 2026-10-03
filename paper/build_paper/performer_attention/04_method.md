<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Performer replaces standard softmax attention with FAVOR+ linear attention:

```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌──────┐       ┌──────┐      ┌──────┐
    │  Q   │       │  K   │      │  V   │
    └──┬───┘       └──┬───┘      └──┬───┘
       │              │              │
       ▼              ▼              │
   ┌───────────────────────┐         │
   │ Feature map φ(·)      │         │
   │ (FAVOR+)              │         │
   └─────────┬─────────────┘         │
             │                       │
             ▼                       │
   ┌───────────────────────┐         │
   │ Linear attention      │◄────────┘
   │ (associative or causal)│
   └─────────┬─────────────┘
             │
             ▼
        Output Y
```

## 4.2 The Feature Map

### 4.2.1 Orthogonal Random Matrix

We build $\omega \in \mathbb{R}^{F \times d}$ in blocks of $d$ rows:

```python
def _build_orthogonal_matrix(d, m):
    blocks = []
    remaining = m
    while remaining > 0:
        block_size = min(d, remaining)
        g = torch.randn(d, block_size)
        q, _ = torch.linalg.qr(g)
        blocks.append(q)
        remaining -= block_size
    W = torch.cat(blocks, dim=1) * math.sqrt(d)
    return W.t().contiguous()
```

Each block is orthonormal. The scaling $\sqrt{d}$ normalizes the variance.

### 4.2.2 FAVOR+ Feature Map

```python
def _feature_map(self, x):
    proj = torch.matmul(x, self.omega.t())
    diag_sq = torch.sum(x ** 2, dim=-1, keepdim=True) * 0.5
    max_proj = proj.amax(dim=-1, keepdim=True).detach()
    diff = torch.clamp(proj - max_proj - diag_sq, min=-30.0, max=10.0)
    phi = torch.exp(diff) + self.eps
    phi = phi / (math.sqrt(self.num_features) + self.eps)
    return phi
```

**Three stabilizations:**
1. **Max-subtraction:** `proj - max_proj` keeps the exponent bounded.
2. **Clamp:** `[-30, 10]` prevents extreme values.
3. **Epsilon:** `+eps` ensures positivity.

## 4.3 Bidirectional Path (Encoders)

For non-causal attention:

$$
KV = \phi(K)^\top V, \quad Z = \sum_j \phi(K_j)
\tag{4.1}
$$

$$
O_i = \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z + \epsilon}
\tag{4.2}
$$

**Masking.** We mask both K and Q features:

```python
K_feat = self._feature_map(K) * mask_expanded
Q_feat = self._feature_map(Q) * mask_expanded
```

Masking Q prevents padding tokens from contributing to the output.

## 4.4 Causal Path (Decoders)

For causal attention, we use a **chunked prefix scan** with `chunk_size = 64`:

For each chunk:

1. **Intra-chunk:** Compute causal attention within the chunk:

$$
\text{intra}_i = \sum_{j \leq i, j \in \text{chunk}} \phi(q_i)^\top \phi(k_j) v_j
\tag{4.3}
$$

2. **Inter-chunk:** Use running state $S, z$:

$$
\text{inter}_i = \phi(q_i)^\top S
\tag{4.4}
$$

3. **Output:**

$$
O_i = \frac{\text{intra}_i + \text{inter}_i}{\text{intra-den}_i + \text{inter-den}_i + \epsilon}
\tag{4.5}
$$

4. **State update:**

$$
S \mathrel{+}= \phi(K_c)^\top V_c, \quad z \mathrel{+}= \sum_j \phi(K_{c,j})
\tag{4.6}
$$

## 4.5 Complexity

| Path | Time | Memory |
|---|---|---|
| Bidirectional | $\mathcal{O}(L F d)$ | $\mathcal{O}(F d)$ |
| Causal (chunked) | $\mathcal{O}(L F d)$ | $\mathcal{O}(F d)$ |
| Baseline softmax | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |

**Linear in $L$.**

## 4.6 Parameter Analysis

| Component | Params |
|---|---|
| Q/K/V/O projections | $4 d^2$ |
| $\omega$ (frozen) | $F \times d$ |
| LoRA | $4 \cdot 2 r_L d$ |
| `output_scale` | $4$ (one per LoRA) |

**Non-LoRA addition:** $F \times d$. For $d = 64$, $F = 64$: 4,096 params.

## 4.7 Two Code/Code-Doc Discrepancies

### 4.7.1 `inject_performer` Raises Instead of Returning None

Docstring says: `Safe injection returning None on failure`.

Code:

```python
except Exception as e:
    logger.warning("PerformerAttention construction/weight copying failed: %s", e)
    raise RuntimeError(f"Injection failed")  # ← raises
```

**Impact.** In production, this crashes the pipeline rather than skipping 
gracefully. Same bug as NASC.

### 4.7.2 `LoRALinear.output_scale` is New

```python
self.output_scale = nn.Parameter(torch.ones(1))
# ...
return out * self.output_scale
```

This parameter is not in the baseline checkpoint. On `load_state_dict`, it 
shows as `missing_key`. Defaults to 1.0 (harmless), but its presence breaks 
`strict=True` loading.

## 4.8 Algorithm

**Algorithm 1:** PerformerAttention forward
```
Input:  X ∈ R^{B×L×d}, is_decoder
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
 2: Q ← Q * d^{-0.25};  K ← K * d^{-0.25}
 3: if RoPE: (Q, K) ← Rotary(Q, K)
 4: if GQA: (K, V) ← repeat_kv(K, V)
 5:
 6: Q_feat ← φ(Q)
 7: K_feat ← φ(K)
 8:
 9: if is_decoder:
10:     # Chunked prefix scan
11:     for c in range(num_chunks):
12:         intra ← causal_attention(Q_feat_c, K_feat_c, V_c)
13:         inter ← Q_feat_c @ S
14:         S += K_feat_c^T @ V_c
15:         z += sum(K_feat_c)
16:     O ← (intra + inter) / (den + ε)
17: else:
18:     # Associative accumulation
19:     KV ← K_feat^T @ V
20:     Z ← sum(K_feat)
21:     O ← (Q_feat @ KV) / (Q_feat @ Z^T + ε)
22:
23: Y ← out_proj(O)
24: return Y
```

## 4.9 Theoretical Notes

### 4.9.1 FAVOR+ Approximation Bound

Choromanski et al. (2021) prove that FAVOR+ achieves:

$$
\|\text{Attn}_{\text{FAVOR+}} - \text{Attn}_{\text{softmax}}\|_\infty \leq \mathcal{O}\!\left(\frac{1}{\sqrt{F}}\right)
\tag{4.7}
$$

with high probability.

### 4.9.2 Why Positive Features?

Softmax weights are non-negative. Standard RFF can produce negative 
weights, breaking the invariant. FAVOR+ uses only positive features.

### 4.9.3 Causality Preserved

The chunked scan preserves strict causality: intra-chunk uses causal mask, 
inter-chunk uses only previously-processed state.

---