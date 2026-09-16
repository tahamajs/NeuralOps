# 4. Method

## 4.1 Overview

VSpec++ replaces standard attention with:

1. **Token routing** — select top-$k$ tokens.
2. **SVD projections** — factorized Q/K/V/O.
3. **Spectral gating** — variational weighting of singular values.
4. **Head gating** — per-example head importance.
5. **LoRA recovery** — trainable adapters on each projection.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
              ▼
      ┌───────────────┐
      │ Token Router  │
      │ (top-k + CLS) │
      └───────┬───────┘
              │ X_routed (B×k×d)
       ┌──────┼──────┐
       ▼      ▼      ▼
    ┌──────┐ ┌──────┐ ┌──────┐
    │Pq(gq)│ │Pk(gk)│ │Pv(gv)│  ← SVD + Spectral Gate
    └──┬───┘ └──┬───┘ └──┬───┘
       │        │        │
       ▼        ▼        ▼
       Q        K        V
       │        │        │
       └────────┼────────┘
                │
                ▼
        Attention × HeadGate(h)
                │
                ▼
          Po(go)  ← SVD + Spectral Gate
                │
                ▼
          Scatter back to L
                │
                ▼
             Output Y
```

## 4.2 Token Router

**Saliency network.**

$$
s_i = \sigma(f_\theta(x_i)), \quad f_\theta: \mathbb{R}^d \to [0, 1]
\tag{4.1}
$$

where $f_\theta$ is a 2-layer MLP: Linear($d$, 32) → GELU → Linear(32, 1).

**Masking.** Padding tokens get $s_i = -10^4$; `[CLS]` gets $s_0 = 10^4$.

**Selection.**

$$
k = \max(k_\text{min}, \lceil \rho L \rceil)
\tag{4.2}
$$

$$
\mathcal{S} = \text{Sort}(\text{TopK}(s, k))
\tag{4.3}
$$

**Gather.** $X_\text{routed} = \text{gather}(X, \mathcal{S})$.

**Nested encoder bypass.** For RoBERTa/BERT, routing is skipped (context 
must be preserved).

## 4.3 SVD-Initialized Projector

For each projection $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

**Step 1: Truncated SVD.**

$$
W \approx U_r \Sigma_r V_r^\top
\tag{4.4}
$$

**Step 2: Store factors.**

$U_r \in \mathbb{R}^{d_\text{out} \times r}$, $S_r \in \mathbb{R}^r$, 
$V_r \in \mathbb{R}^{d_\text{in} \times r}$.

**Step 3: Forward with spectral gate.**

$$
z = (x V_r) \cdot (S_r \odot g)
\tag{4.5}
$$

$$
y = z U_r^\top + b + \text{LoRA}(x)
\tag{4.6}
$$

where $g \in [0, 1]^r$ is the spectral gate (see §4.4).

## 4.4 Variational Spectral Gate

**Pool.** Masked mean over routed tokens:

$$
\bar{x} = \frac{\sum_i m_i x_i}{\sum_i m_i + \epsilon}
\tag{4.7}
$$

**Network.**

$$
[\mu, \log \sigma] = f_\text{sg}(\bar{x}) \in \mathbb{R}^{2r}
\tag{4.8}
$$

where $f_\text{sg}$ is a 2-layer MLP: Linear($d$, 64) → GELU → Linear(64, $2r$).

**Clamp.** $\log \sigma \in [-10, 2]$.

**Sample (training).**

$$
\epsilon \sim \mathcal{N}(0, I), \quad z = \mu + \exp(\log \sigma) \odot \epsilon
\tag{4.9}
$$

**Gate.**

$$
g = \sigma(z) \in [0, 1]^r
\tag{4.10}
$$

**Inference.** $g = \sigma(\mu)$ (deterministic).

**Initialization.** We set last-layer weight to 0, bias to $[\mathbf{3}, \mathbf{-3}]$, 
giving $g \approx \sigma(3) \approx 0.95$ and low variance. This means the 
SVD representation is preserved at step 0.

## 4.5 Head Gate

**Pool.**

$$
\bar{x} = \text{masked-mean}(X_\text{routed})
\tag{4.11}
$$

**Network.**

$$
h = \sigma(f_h(\bar{x})) \in [0, 1]^H
\tag{4.12}
$$

where $f_h$ is Linear($d$, 32) → GELU → Linear(32, $H$).

**Application.** Multiply attention logits:

$$
\text{scores} \leftarrow \text{scores} \cdot h_i
\tag{4.13}
$$

where $h_i$ is the importance of head $i$.

## 4.6 Combined Attention

$$
Q = P_q(X_\text{routed}, g_q), \quad K = P_k(X_\text{routed}, g_k), \quad V = P_v(X_\text{routed}, g_v)
\tag{4.14}
$$

Reshape to multi-head, apply RoPE, GQA, and:

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d_h}} \cdot h
\tag{4.15}
$$

$$
A = \text{softmax}(\text{scores} + \text{causal} + \text{pad})
\tag{4.16}
$$

$$
C = A V
\tag{4.17}
$$

$$
Y_\text{routed} = P_o(C, g_q)
\tag{4.18}
$$

**Scatter back.** For non-routed tokens, we set $Y = 0$ and rely on the 
residual connection to preserve them:

$$
Y = \text{scatter}(Y_\text{routed}, \mathcal{S})
\tag{4.19}
$$

## 4.7 Algorithm

**Algorithm 1:** VSpec++ forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×L×d}

 1: if nested_encoder: S ← arange(L); k ← L
 2: else:              s ← saliency(X)
 3:                    s[0] ← 1e4; s[padding] ← -1e4
 4:                    S ← Sort(TopK(s, k))
 5: X_r ← gather(X, S)
 6: x̄ ← masked_mean(X_r)
 7: for proj in [q, k, v, o]:
 8:     μ_proj, log σ_proj ← MLP(x̄)
 9:     z_proj ← μ_proj + exp(log σ_proj) ⊙ ε   ▷ training
10:     g_proj ← σ(z_proj)
11: h ← σ(MLP(x̄))                               ▷ head gate
12: Q ← Pq(X_r, g_q);  K ← Pk(X_r, g_k);  V ← Pv(X_r, g_v)
13: Q, K ← RoPE(Q, K, pos)
14: if GQA: K, V ← repeat_kv
15: scores ← Q K^T / √d_h × h
16: A ← softmax(scores + masks)
17: C ← A V
18: Y_r ← Po(C, g_q)
19: Y ← scatter(Y_r, S)
20: return Y
```

## 4.8 Complexity Analysis

Per block:

| Component | Time | Params |
|---|---|---|
| Router | $O(L d)$ | $O(d)$ |
| Spectral gates (×3) | $O(d \cdot 2r)$ | $O(dr)$ |
| Head gate | $O(d H)$ | $O(dH)$ |
| Q/K/V proj | $O(k d r)$ | $3 \cdot 2 d r$ |
| Attention | $O(k^2 d)$ | 0 |
| Out proj | $O(k d r)$ | $2 d r$ |
| LoRA | $O(k d r_L)$ | $4 d r_L$ |
| **Total** | $O(L d + k^2 d)$ | $\sim 8 r d + O(dH)$ |

For $d = 768$, $r = 128$, $H = 12$, $\rho = 0.75$:

| Component | Params |
|---|---|
| SVD factors (Q/K/V/O) | $4 \times 2 \times 768 \times 128 = 786K$ |
| Gates | ~150K |
| LoRA | ~4 × 16 × 768 × 2 = 98K |
| Router | ~25K |
| **Total** | ~1.06M |

vs baseline $4 d^2 = 2.36M$ → **2.2× reduction**.

## 4.9 Training

Trainable:
- Router MLP.
- Spectral gates (3 × MLP).
- Head gate MLP.
- LoRA adapters ($A, B$ on each projection).

Frozen:
- SVD factors $U, S, V$.
- Bias (optional).

**Variational penalty.** In practice, we add a small KL-like regularization:

$$
\mathcal{L}_\text{KL} = \beta \sum_i \left(\mu_i^2 + \sigma_i^2 - 1\right)
\tag{4.20}
$$

to encourage low-variance gates.

## 4.10 Inference

At inference, the variational sampling is replaced by $z = \mu$. This gives 
a **deterministic** forward pass, matching the training-time mean of the 
posterior.

---