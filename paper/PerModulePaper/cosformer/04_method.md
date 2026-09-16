<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The implementation
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Our CosformerAttention module has two computational paths:

1. **Encoder (bidirectional):** associative accumulation.
2. **Decoder (causal):** chunked prefix-sum.

Both share the same feature map $\phi(\cdot) = \text{ELU} + 1$ (or softplus) 
and cosine/sine reweighting.

```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌──────┐       ┌──────┐      ┌──────┐
    │  Q   │       │  K   │      │  V   │
    │ proj │       │ proj │      │ proj │
    └──┬───┘       └──┬───┘      └──┬───┘
       │              │              │
       │              │              │
       ▼              ▼              │
   ┌───────────────────────┐         │
   │  φ(·) = ELU+1         │         │
   └─────────┬─────────────┘         │
             │                       │
             ▼                       │
   ┌───────────────────────┐         │
   │  Cos/Sin reweighting  │         │
   └─────────┬─────────────┘         │
             │                       │
             ├───────────┬───────────┤
             │           │           │
             ▼           ▼           ▼
        ┌────────────────────────────────┐
        │ is_decoder?                    │
        └──┬─────────────────────────┬───┘
           │ No                      │ Yes
           │                         │
           ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Associative  │          │ Chunked      │
    │ accumulation │          │ prefix-sum   │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
                   Output Y
```

## 4.2 Feature Map

We support two feature maps:

$$
\phi(x) = \begin{cases}
\text{elu}(x) + 1 & \text{(default)} \\
\text{softplus}(x) & \text{(alternative)}
\end{cases}
\tag{4.1}
$$

Both are non-negative, deterministic, and cheap.

**Why not ReLU?** ReLU has zero derivative for negative inputs, which can 
cause dead neurons. ELU + 1 has a non-zero gradient everywhere.

**Why not RFF?** RFF is stochastic and requires more features for the same 
accuracy. ELU + 1 is deterministic and works well in practice.

## 4.3 Positional Reweighting

For sequence length $N$ and max position $M = \max(\text{max\_seq\_len}, N)$:

$$
\theta_i = \frac{\pi i}{2 M}, \quad i = 0, 1, \dots, N-1
\tag{4.2}
$$

The cosine and sine factors are:

$$
c_i = \cos(\theta_i), \quad s_i = \sin(\theta_i)
\tag{4.3}
$$

Applied to $Q, K$:

$$
\tilde{Q} = Q \odot c, \quad \tilde{Q}' = Q \odot s, \quad
\tilde{K} = K \odot c, \quad \tilde{K}' = K \odot s
\tag{4.4}
$$

Broadcasting: $c, s \in \mathbb{R}^{1 \times 1 \times N \times 1}$, 
$Q, K \in \mathbb{R}^{B \times H \times N \times d}$.

## 4.4 Encoder Path: Associative Accumulation

For bidirectional attention:

$$
\begin{aligned}
KV_c &= \tilde{K}^\top V \in \mathbb{R}^{B \times H \times d \times d} \\
KV_s &= \tilde{K}'^\top V \in \mathbb{R}^{B \times H \times d \times d} \\
Z_c &= \sum_j \tilde{K}_j \in \mathbb{R}^{B \times H \times 1 \times d} \\
Z_s &= \sum_j \tilde{K}'_j \in \mathbb{R}^{B \times H \times 1 \times d}
\end{aligned}
\tag{4.5}
$$

The output:

$$
\text{num} = \tilde{Q} KV_c + \tilde{Q}' KV_s, \quad
\text{den} = \tilde{Q} Z_c^\top + \tilde{Q}' Z_s^\top + \epsilon
\tag{4.6}
$$

$$
\text{context} = \frac{\text{num}}{\text{den}}
\tag{4.7}
$$

**Complexity.** Each matmul is $\mathcal{O}(L d^2)$ or $\mathcal{O}(d^2 L)$. 
Total: $\mathcal{O}(L d^2)$ time, $\mathcal{O}(d^2)$ memory for the state.

## 4.5 Decoder Path: Chunked Prefix-Sum

For causal attention:

1. Pad sequence to multiple of `chunk_size = 32`.
2. Split into `num_chunks = L / 32` chunks.
3. For each chunk:
   - Compute **intra-chunk** scores (with causal mask).
   - Compute **inter-chunk** contributions via running state $(S, Z)$.
   - Append output.
   - Update state.

**Intra-chunk:**

$$
\text{num}_\text{intra} = \text{masked}(\tilde{Q}_c \tilde{K}_c^\top) V_c
\tag{4.8}
$$

**Inter-chunk:**

$$
\text{num}_\text{inter} = \tilde{Q}_c S_c + \tilde{Q}_c' S_s
\tag{4.9}
$$

**State update:**

$$
S_c \mathrel{+}= \tilde{K}_c^\top V_c, \quad
Z_c \mathrel{+}= \tilde{K}_c^\top \mathbf{1}
\tag{4.10}
$$

The output for chunk $c$:

$$
\text{context}_c = \frac{\text{num}_\text{intra} + \text{num}_\text{inter}}{\text{den}_\text{intra} + \text{den}_\text{inter} + \epsilon}
\tag{4.11}
$$

**Complexity.** Outer Python loop over `num_chunks` iterations (4 for $L = 128$). 
Each iteration does a small matmul. Total wall-clock: dominated by 4 kernel 
launches per chunk × 2 chunks.

## 4.6 Weight Transfer

### 4.6.1 Standard (DistilBERT)

```python
q_w = src.q_lin.weight.data
k_w = src.k_lin.weight.data
v_w = src.v_lin.weight.data
o_w = src.out_lin.weight.data
```

### 4.6.2 Nested (RoBERTa)

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 4.6.3 Fused (GPT-2)

```python
c_w = src.c_attn.weight.data.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.data.t()
```

## 4.7 Parameter Analysis

| Component | Baseline | Cosformer |
|---|---|---|
| Q projection | $d^2$ | $d^2$ + LoRA |
| K projection | $d^2$ | $d^2$ + LoRA |
| V projection | $d^2$ | $d^2$ + LoRA |
| O projection | $d^2$ | $d^2$ + LoRA |
| **Per block** | $4d^2$ | $4d^2 + 4 \cdot r(d + d)$ |

**Example (DistilBERT, $r = 16$, $d = 768$):**

Per block: $4 \cdot 768^2 + 4 \cdot 16 \cdot 1536 = 2{,}359{,}296 + 98{,}304 
= 2{,}457{,}600$.

Δ% per block: $+4.17\%$.

**Cosformer does not reduce parameter count.**

## 4.8 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Encoder | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d^2)$ |
| Decoder | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d^2)$ |
| Baseline MHSA | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |

**For $L = 128$, $d = 64$:**
- Cosformer: ~128 × 64² = 524K operations per head.
- Baseline: ~128² × 64 = 1M operations per head.
- **Cosformer has ~2× fewer FLOPs.**

**Wall-clock:** Cosformer has overhead from:
- Kernel feature map (ELU)
- Cos/sin reweighting
- Chunked scan (decoder only)

## 4.9 Algorithm

**Algorithm 1:** CosformerAttention forward
```
Input:  X ∈ R^{B×L×d}, is_decoder, chunk_size C
Output: Y ∈ R^{B×L×d}

1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
2: if RoPE: (Q, K) ← Rotary(Q, K)
3: if GQA: (K, V) ← repeat_kv(K, V)
4:
5: # Feature map
6: Q ← ELU(Q) + 1
7: K ← ELU(K) + 1
8:
9: # Positional reweighting
10: θ ← π * arange(L) / (2 * max_seq_len)
11: Q_c ← Q ⊙ cos(θ); Q_s ← Q ⊙ sin(θ)
12: K_c ← K ⊙ cos(θ); K_s ← K ⊙ sin(θ)
13:
14: if is_decoder:
15:     Y ← chunked_prefix_sum(Q_c, Q_s, K_c, K_s, V, C)
16: else:
17:     KV_c ← K_c^T @ V
18:     KV_s ← K_s^T @ V
19:     Z_c ← sum(K_c); Z_s ← sum(K_s)
20:     num ← Q_c @ KV_c + Q_s @ KV_s
21:     den ← Q_c @ Z_c^T + Q_s @ Z_s^T
22:     Y ← num / (den + ε)
23:
24: if not is_nested_encoder:
25:     Y ← out_proj(Y)
26: return Y
```

## 4.10 Theoretical Notes

### 4.10.1 Kernel Approximation

ELU + 1 is **not** a good approximation of $\exp(q^\top k)$. The 
approximation error is bounded by:

$$
\left|\phi(q)^\top \phi(k) - e^{q^\top k}\right| \leq \epsilon_\text{ELU}
\tag{4.12}
$$

where $\epsilon_\text{ELU}$ grows with $\|q\|$ and $\|k\|$.

**Why it works anyway.** The **softmax is scale-invariant** — only the 
relative ordering of $q^\top k$ matters. ELU + 1 preserves the ordering 
approximately.

### 4.10.2 Locality Recovery

The Cosformer reweighting gives:

$$
\tilde{q}_i^\top \tilde{k}_j + \tilde{q}_i'^\top \tilde{k}_j' = \phi(q_i)^\top \phi(k_j) \cdot \cos(\theta_i - \theta_j)
\tag{4.13}
$$

The cosine term decays from 1 at $|i - j| = 0$ to 0 at $|i - j| = L$ 
(approximately). This restores locality.

### 4.10.3 Causality

The causal path preserves causality:

$$
\frac{\partial Y_i}{\partial X_j} = 0 \text{ for } j > i
\tag{4.14}
$$

because intra-chunk scores are masked and inter-chunk uses only 
previously-processed state.

---