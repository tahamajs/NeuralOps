<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

NASC has two paths, chosen by the flag `use_dense`:

```python
use_dense = is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders)
```

- **Cross-attention:** dense (standard softmax).
- **Encoders:** dense by default (unless `enable_cascade_on_encoders=True`).
- **Decoders:** cascade.

The cascade path combines three branches:

```
                    Input X
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
    ┌──────┐       ┌──────┐       ┌──────┐      ┌──────┐
    │  Q   │       │  K   │       │  V   │      │ Gate │
    └──┬───┘       └──┬───┘       └──┬───┘      └──┬───┘
       │              │              │              │
       └──────┬───────┴──────────────┤              │
              │                      │              │
   ┌──────────┼───────────┐          │              │
   │          │           │          │              │
   ▼          ▼           ▼          ▼              ▼
 ┌─────┐   ┌──────┐  ┌────────┐   ┌──────┐      [g₀,g₁,g₂]
 │Local│   │Block │  │RFF     │   │      │       softmax
 │Win  │   │Sparse│  │Global  │   │      │       per-batch
 └──┬──┘   └──┬───┘  └───┬────┘   │      │
    │         │          │         │      │
    └─────────┴──────────┴─────────┘      │
              │                            │
              ▼                            ▼
        g₀·O_loc + g₁·O_sp + g₂·O_glb
              │
              ▼
        Output Y
```

## 4.2 Local Window Branch

Standard scaled dot-product attention with window mask:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
\tag{4.1}
$$

Mask:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W \text{ and } j \leq i \text{ (if causal)} \\ 0 & \text{otherwise} \end{cases}
\tag{4.2}
$$

Standard softmax attention within the window.

## 4.3 Sparse Block Branch

### 4.3.1 Block-Level Scores

Partition into $n_b = \lceil L / B_s \rceil$ blocks. Compute mean Q and K:

$$
\bar{q}_i = \text{mean}(Q_{\text{block } i}), \quad
\bar{k}_j = \text{mean}(K_{\text{block } j})
\tag{4.3}
$$

Block scores:

$$
s_{ij} = \frac{\bar{q}_i^\top \bar{k}_j}{\sqrt{d}}
\tag{4.4}
$$

### 4.3.2 Top-$k$ Selection

For each query block $i$, select top-$k$ key blocks:

$$
\mathcal{B}_i = \text{topk}_{k}(\{s_{ij}\}_j)
\tag{4.5}
$$

### 4.3.3 Causal Handling

For causal decoders, blocks $j > i$ are masked. Also, block 0 is always 
selected (an attention sink).

### 4.3.4 Attention Within Selected Blocks

Concatenate the selected blocks' K, V, compute softpick attention:

$$
O_{\text{sp}, i} = \text{softpick}(Q_i K_{\mathcal{B}_i}^\top / \sqrt{d}) V_{\mathcal{B}_i}
\tag{4.6}
$$

**Implementation note.** The block loop uses a Python `for` loop over 
$n_b$ iterations, gathering selected blocks each time. For $L = 128$, 
$B_s = 32$, $n_b = 4$.

## 4.4 Global Spectral RFF Branch

### 4.4.1 Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\Omega x), \sin(\Omega x)] \in \mathbb{R}^{2F}
\tag{4.7}
$$

where $\Omega \in \mathbb{R}^{d \times F}$ is orthogonally initialized.

### 4.4.2 Non-Causal Path

$$
KV = \phi(K)^\top V, \quad Z = \sum_j \phi(K_j)
\tag{4.8}
$$

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top KV}{\phi(q_i)^\top Z}
\tag{4.9}
$$

### 4.4.3 Causal Path via Prefix Sums

For causal decoders:

$$
KV_c = \text{cumsum}(\phi(K) \otimes V) \in \mathbb{R}^{B \times H \times L \times 2F \times d}
\tag{4.10}
$$

$$
Z_c = \text{cumsum}(\phi(K))
\tag{4.11}
$$

Then per-token:

$$
O_{\text{glb}, i} = \frac{\phi(q_i)^\top KV_{c,i}}{\phi(q_i)^\top Z_{c,i} + \epsilon}
\tag{4.12}
$$

**Memory concern.** $KV_c$ is $B \times H \times L \times 2F \times d$. For 
$B = 8$, $H = 12$, $L = 128$, $F = 64$, $d = 64$: 
$8 \times 12 \times 128 \times 128 \times 64 = 8 \times 10^9$ elements.

**This is 32 GB in FP32.** In practice, PyTorch's cumsum and einsum might 
materialize this, causing OOM on RTX 4070.

## 4.5 Gate

Per-batch gate:

$$
\bar{x}_b = \frac{1}{L} \sum_t x_{b,t}
\tag{4.13}
$$

$$
g = \text{softmax}(W_g \bar{x}_b) \in \mathbb{R}^3
\tag{4.14}
$$

where $W_g \in \mathbb{R}^{3 \times d}$.

**All tokens in a batch share the same $g$.** No per-token adaptivity.

## 4.6 Fusion

$$
O = g_0 \cdot O_{\text{loc}} + g_1 \cdot O_{\text{sp}} + g_2 \cdot O_{\text{glb}}
\tag{4.15}
$$

## 4.7 Parameter Analysis

| Component | Params |
|---|---|
| Q/K/V/O projections | $4 d^2$ |
| RFF $\Omega$ | $d \times F$ (frozen) |
| Gate $W_g$ | $3 d$ |
| LoRA | $4 \cdot 2 r_L d$ |
| **Non-LoRA addition** | $d F + 3d$ |

For $d = 768, F = 64$: RFF = 49,152 + gate = 2,304.

## 4.8 Complexity Analysis

| Branch | Time | Memory |
|---|---|---|
| Local window | $O(L W d)$ | $O(L W)$ |
| Sparse block | $O(L k B_s d)$ | $O(L k B_s)$ |
| RFF (non-causal) | $O(L F d)$ | $O(F d)$ |
| RFF (causal) | $O(L F d)$ | $O(L F d)$ ← high |
| Gate | $O(L d)$ | $O(d)$ |
| **Total (cascade)** | $O(L(W + k B_s + F) d)$ | $O(L F d)$ |

## 4.9 Algorithm

**Algorithm 1:** NASC forward
```
Input:  X ∈ R^{B×L×d}, is_decoder, is_cross, W, B_s, k, F
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q; K ← X W_K; V ← X W_V
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA: (K, V) ← repeat_kv(K, V)
 4:
 5: if is_cross or (not is_decoder and not enable_cascade):
 6:     # Dense path
 7:     O ← softmax(Q K^T / sqrt(d)) V
 8: else:
 9:     # Cascade path
10:     g ← softmax(W_g · mean(X, dim=1))   ▷ per-batch gate
11:     O_loc ← local_window_attention(Q, K, V, W)
12:     O_sp ← sparse_block_attention(Q, K, V, B_s, k)
13:     O_glb ← spectral_rff_attention(Q, K, V, F)
14:     O ← g[0]·O_loc + g[1]·O_sp + g[2]·O_glb
15:
16: if head_mask: apply
17: if is_nested_encoder: return O
18: else: return out_proj(O)
```

---