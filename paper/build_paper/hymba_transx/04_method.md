# 4. Method

## 4.1 Overview

HymbaTransX replaces standard attention with a **two-branch block**:

- **Branch 1: Differential attention** — noise-reduced softmax attention.
- **Branch 2: SimpleSSM** — linear recurrent mixing.

A **learned scalar gate** $\alpha$ blends them.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
       ┌──────┴──────┐
       ▼             ▼
   ┌───────┐    ┌───────┐
   │ Diff  │    │ SSM   │
   │ Attn  │    │       │
   └───┬───┘    └───┬───┘
       │            │
       │ α=σ(w)     │
       │  ┌─────────┘
       ▼  ▼
   α⊙A + (1-α)⊙M
       │
       ▼
     out_proj
       │
       ▼
    Output Y
```

## 4.2 Differential Attention Branch

We compute two independent Q-K pairs:

$$
Q_1 = X W_{Q_1}, \quad K_1 = X W_{K_1}
\tag{4.1}
$$

$$
Q_2 = X W_{Q_2}, \quad K_2 = X W_{K_2}
\tag{4.2}
$$

Two attention maps:

$$
A_1 = \text{softmax}\!\left(\frac{Q_1 K_1^\top}{\sqrt{d_h}}\right), \quad
A_2 = \text{softmax}\!\left(\frac{Q_2 K_2^\top}{\sqrt{d_h}}\right)
\tag{4.3}
$$

Differential output:

$$
A_\text{diff} = \text{ReLU}(A_1 - \lambda A_2), \quad \lambda = 0.2
\tag{4.4}
$$

Then normalize:

$$
\tilde{A}_\text{diff} = \frac{A_\text{diff}}{\sum A_\text{diff} + \epsilon}
\tag{4.5}
$$

Value projection:

$$
V = X W_V, \quad A_\text{out} = \tilde{A}_\text{diff} V
\tag{4.6}
$$

**Why $\lambda = 0.2$?** Ye et al. use $\lambda = 0.8$ for pure differential 
attention. Since we blend with SSM, we soften the damping to retain more 
attention signal.

**Why ReLU?** Prevents negative values from flipping sign of the value 
contribution.

**Why normalize after ReLU?** The subtraction can produce a sparse map with 
very small total mass. Normalization preserves output scale.

## 4.3 SSM Branch

We use a `SimpleSSM` module with element-wise recurrence:

$$
h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t, \quad
y_t = \bar{C}_t \odot h_t
\tag{4.7}
$$

where $\bar{A}_t \in (0, 1)$ is a data-dependent decay.

**Bidirectional handling.** For encoders, we run two SSM passes:

$$
M_\text{fwd} = \text{SSM}(X), \quad
M_\text{bwd} = \text{flip}(\text{SSM}(\text{flip}(X)))
\tag{4.8}
$$

$$
M_\text{ssm} = 0.5 (M_\text{fwd} + M_\text{bwd})
\tag{4.9}
$$

For decoders, only the forward pass is used.

## 4.4 Fusion

A single learnable scalar $\alpha$:

$$
\alpha = \sigma(w), \quad w \in \mathbb{R}
\tag{4.10}
$$

Initialized to $w = -1$, giving $\alpha = 0.27$.

Final output:

$$
\text{combined} = \alpha \odot A_\text{out} + (1 - \alpha) \odot M_\text{ssm}
\tag{4.11}
$$

then projected by `out_proj`.

## 4.5 HeadRoPE Fallback

When the parent model does not pass `position_embeddings`, we compute them 
internally:

$$
\theta_{k, i} = i \cdot \text{base}^{-2k/d_h}, \quad \text{base} = 10^4
\tag{4.12}
$$

$$
\text{cos}_{i, k} = \cos(\theta_{k, i}), \quad \text{sin}_{i, k} = \sin(\theta_{k, i})
\tag{4.13}
$$

Then apply:

$$
\tilde{Q}_i = Q_i \cos_i + \text{rot}(Q_i) \sin_i
\tag{4.14}
$$

**Cache.** Cos/sin tables are cached up to `max_seq_len = 4096` and extended 
on demand.

## 4.6 Weight Transfer

Our `copy_weights` handles four patterns:

| Architecture | Source | Target |
|---|---|---|
| DistilBERT | `q_lin/k_lin/v_lin/out_lin` | `W_q1/W_k1/W_v/out_proj` |
| RoBERTa | `self.query/key/value` + `output.dense` | same |
| GPT-2 | `c_attn` (fused, transposed) + `c_proj` | same |
| Qwen3.5 | `q_proj/k_proj/v_proj/o_proj` | same |

$W_{Q_2}$ and $W_{K_2}$ are initialized independently via Kaiming uniform — 
they have no pretrained counterpart.

**Fallback on missing weights.** If a projection is missing after extraction, 
we log a warning and keep the Kaiming-initialized default. This is a 
**soft fallback** (unlike SOA Fused which raises).

## 4.7 Algorithm

**Algorithm 1:** HymbaTransX forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
        position_embeddings (optional)
Output: Y ∈ R^{B×L×d}

 1: Q1 ← W_q1(X);  K1 ← W_k1(X)
 2: Q2 ← W_q2(X);  K2 ← W_k2(X)
 3: V  ← W_v(X)
 4: if pos_emb: Q_i, K_i ← RoPE(Q_i, K_i, pos_emb)
 5: else:       Q_i, K_i ← HeadRoPE(Q_i, K_i)
 6: if GQA: K_i, V ← repeat_kv(...)
 7: A1 ← softmax(Q1 K1^T / √d_h)
 8: A2 ← softmax(Q2 K2^T / √d_h)
 9: if causal: mask A1, A2
10: A_diff ← ReLU(A1 - λ A2)
11: A_norm ← A_diff / (sum + ε)
12: A_out ← A_norm @ V
13: if decoder: M ← SSM(X)
14: else:       M ← 0.5(SSM(X) + flip(SSM(flip(X))))
15: α ← sigmoid(w)
16: out ← α A_out + (1-α) M
17: Y ← out_proj(out)
18: return Y
```

## 4.8 Complexity Analysis

For a single block:

| Component | Time | Params |
|---|---|---|
| $W_{Q_1}, W_{K_1}$ | $O(L d^2)$ | $2 d^2$ |
| $W_{Q_2}, W_{K_2}$ | $O(L d^2)$ | $2 d^2$ |
| $W_V$ | $O(L d^2)$ | $d^2$ |
| Diff attention | $O(L^2 d)$ | 0 |
| SSM | $O(L d)$ | $2 d^2$ |
| $W_O$ | $O(L d^2)$ | $d^2$ |
| **Total** | $O(L^2 d)$ | $8 d^2$ |

Compared to standard attention ($4 d^2$ params), HymbaTransX has **$8 d^2$** — 
double the parameter count.

## 4.9 Parameter Analysis

For $d = 768$, per layer:

| Component | Baseline | HymbaTransX |
|---|---|---|
| Q, K, V, O | 4 × 589,824 = 2.36M | — |
| W_q1, W_k1 | — | 1.18M |
| W_q2, W_k2 | — | 1.18M |
| W_v | — | 0.59M |
| SSM | 0 | 0.59M |
| W_o | — | 0.59M |
| **Total** | **2.36M** | **4.13M** |
| **Increase** | — | **+75%** |

## 4.10 Training

Trainable:
- LoRA adapters on all projections.
- $W_{Q_2}, W_{K_2}$ (Kaiming init, then LoRA-tuned).
- SSM parameters.
- $\alpha$ (scalar).

Frozen:
- Base Q/K/V/O weights (transferred from pretrained attention).

**Trainable count:** ~0.8M per layer.

---