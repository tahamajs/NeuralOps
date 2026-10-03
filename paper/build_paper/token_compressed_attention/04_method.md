# 4. Method

## 4.1 Overview

TokenCompressedAttention (TCA) replaces standard attention by adding three 
steps before the attention computation:

1. **Score computation** (§4.2): compute per-token importance.
2. **Token selection** (§4.3): keep the top-$k$ tokens.
3. **Causal-aware attention** (§4.4): apply attention with a mask over 
   original positions.

The overall pipeline:

```
    Input X ∈ R^{B×L×d}
              │
              ▼
    ┌───────────────────────┐
    │ Project Q/K/V         │  (frozen base weights + LoRA)
    └───────────┬───────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
    ┌──────┐         ┌──────────────┐
    │ Q    │         │ K, V         │
    │      │         │ Score K.V    │
    │      │         └──────┬───────┘
    │      │                │
    │      │         ┌──────▼───────┐
    │      │         │ Top-k select │
    │      │         │ [CLS] pinned │
    │      │         └──────┬───────┘
    │      │                │
    └──────┴────────────────┘
              │
              ▼
    ┌───────────────────────┐
    │ Attention + causal    │
    │ mask against original │
    │ positions             │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │ Output projection     │
    └───────────────────────┘
```

## 4.2 Token Importance Scoring

For each token $i$, we compute:

$$
s_i = \underbrace{\|x_i\|_2}_{\text{energy}} + \underbrace{f_\theta(x_i)}_{\text{learned}}
\tag{4.1}
$$

where:

- $\|x_i\|_2$ is the $\ell_2$ norm of the hidden state.
- $f_\theta$ is a small MLP: Linear($d$, 64) → GELU → Linear(64, 1).

**Why this combination?**

- The norm captures **activation energy** — an established proxy for token 
  importance (Lin et al., 2023).
- The learned term captures **content-specific** importance that a 
  norm-based heuristic misses.

## 4.3 Token Selection

We force the first token to be retained (attention sink):

$$
s_0 \leftarrow 10^4
\tag{4.2}
$$

We mask out padding tokens:

$$
s_i \leftarrow -10^4 \quad \text{for padding tokens}
\tag{4.3}
$$

Then we select the top-$k$ indices:

$$
\mathcal{S} = \text{TopK}(s, k), \quad k = \lceil \rho L \rceil
\tag{4.4}
$$

and sort them to preserve order:

$$
\mathcal{S} \leftarrow \text{Sort}(\mathcal{S})
\tag{4.5}
$$

Finally, we gather:

$$
K_c = K[\mathcal{S}], \quad V_c = V[\mathcal{S}]
\tag{4.6}
$$

**Short-sequence handling.** If $L \leq 2$, no compression is applied.

## 4.4 Causal-Safe Attention

Standard attention over the compressed keys:

$$
\text{scores}_{i, j'} = \frac{q_i \cdot K_{c, j'}^\top}{\sqrt{d_h}}
\tag{4.7}
$$

For decoders, we add a mask based on **original positions**:

$$
\text{mask}_{i, j'} = \begin{cases} 0 & \mathcal{S}_{j'} \leq i \\ -\infty & \mathcal{S}_{j'} > i \end{cases}
\tag{4.8}
$$

This ensures that query at position $i$ cannot attend to compressed keys 
whose original position exceeds $i$.

**Computing the mask:**

```python
q_pos = torch.arange(L).view(1, 1, L, 1)  # (1, 1, L, 1)
k_orig_pos = top_idx.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k)
causal_mask = k_orig_pos > q_pos  # (B, 1, L, k)
scores = scores.masked_fill(causal_mask, -inf)
```

## 4.5 Combined Algorithm

**Algorithm 1:** TCA forward pass
```
Input:  hidden_states X ∈ R^{B×L×d}
        attention_mask ∈ {0,1}^{B×L}
        position_ids
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
 2: Q, K ← RoPE(Q, K, position_ids)
 3: s ← ‖X‖_2 + score_net(X)        ▷ Per-token score
 4: s[0] ← 1e4                       ▷ Pin [CLS]
 5: s[padding] ← -1e4                ▷ Mask padding
 6: S ← TopK(s, k)                   ▷ Select top-k
 7: S ← Sort(S)                      ▷ Order-preserve
 8: K_c ← K[S];  V_c ← V[S]          ▷ Gather
 9: K_c ← repeat_kv(K_c, H/H_kv)     ▷ GQA expand
10: scores ← Q @ K_c^T / √d_h
11: if is_decoder:
12:     mask ← (S[None,None,None,:] > arange(L)[None,None,:,None])
13:     scores ← scores.masked_fill(mask, -inf)
14: scores ← mask_padding(scores, attention_mask)
15: A ← softmax(scores)
16: Y ← A @ V_c
17: Y ← out_proj(Y)
18: return Y
```

## 4.6 Memory and Compute Savings

For a single attention layer:

| Metric | Baseline | TCA | Savings |
|---|---|---|---|
| KV cache | $2 H_{kv} L d_h$ | $2 H_{kv} k d_h$ | $1 - \rho$ |
| Attention compute | $L^2$ | $L \cdot k$ | $1 - \rho$ |
| Score net compute | 0 | $2 L \cdot (d \cdot 64 + 64)$ | +small |

For $\rho = 0.5$:

- KV cache: **50% reduction**
- Attention compute: **50% reduction**
- Score net: **~0.5% overhead**

## 4.7 Parameter Analysis

**Important: TCA does NOT reduce parameter count.**

| Component | Baseline | TCA |
|---|---|---|
| Projections | $4 d^2$ (frozen) | $4 d^2$ (frozen) |
| LoRA | 0 | $4 \cdot r_L (d + d)$ |
| Score net | 0 | $d \cdot 64 + 64 + 64 \cdot 1 + 1$ |
| **Total per layer** | $4 d^2$ | $4 d^2 + 8 r_L d + 65 d$ |

For $d = 768$, $r_L = 16$:

| Component | Params |
|---|---|
| Baseline | 2,359,296 |
| TCA | 2,359,296 + 98,304 + 49,920 = **2,507,520** |
| **Change** | **+6.3%** |

**We are explicit about this in the abstract.**

## 4.8 Compression Ratio Selection

The compression ratio $\rho$ trades off:

| $\rho$ | KV reduction | Accuracy drop |
|---|---|---|
| 0.9 | 10% | ~0% |
| 0.7 | 30% | ~1% |
| 0.5 | 50% | ~3% |
| 0.3 | 70% | ~8% |
| 0.1 | 90% | severe |

We recommend $\rho = 0.5$ as a default.

---