<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

GQA is a **structural modification** of the attention mechanism: it reduces 
the number of $K, V$ heads from $H$ to $G$ while keeping all $H$ query heads 
intact. The modification consists of four components:

1. **Group configuration** (§4.2): choose $G$.
2. **Projection initialization** (§4.3): initialize $K, V$ for the reduced head count.
3. **Weight transfer** (§4.4): copy query weights verbatim; group-reduce KV weights.
4. **Forward pass** (§4.5): apply standard attention with KV repetition.

**Figure 4.1** shows the difference between MHA, GQA, and MQA at $H = 8$.

```
    MHA (H_KV = 8)          GQA (H_KV = 4)          MQA (H_KV = 1)

    Q1 ── K1 V1             Q1 ── K1 V1             Q1 ── K1 V1
    Q2 ── K2 V2             Q2 ── ┘                 Q2 ── ┘
    Q3 ── K3 V3             Q3 ── K2 V2             Q3 ── ┘
    Q4 ── K4 V4             Q4 ── ┘                 Q4 ── ┘
    Q5 ── K5 V5             Q5 ── K3 V3             Q5 ── ┘
    Q6 ── K6 V6             Q6 ── ┘                 Q6 ── ┘
    Q7 ── K7 V7             Q7 ── K4 V4             Q7 ── ┘
    Q8 ── K8 V8             Q8 ── ┘                 Q8 ── ┘
```

## 4.2 Group Configuration

The number of KV heads $G$ satisfies $1 \le G \le H$. Two natural choices:

- **Default:** $G = H / 2$ (2× KV cache reduction).
- **Aggressive:** $G = H / 4$ or $G = 1$ (MQA).

Our default is $G = H / 2$ to balance quality and memory.

## 4.3 Projection Initialization

For a target model with hidden size $d$ and per-head dimension $d_h$:

$$
W_Q \in \mathbb{R}^{d \times H d_h}, \qquad
W_K, W_V \in \mathbb{R}^{d \times G d_h}, \qquad
W_O \in \mathbb{R}^{H d_h \times d}
\tag{4.1}
$$

The query and output projections have unchanged shapes; only $W_K, W_V$ 
shrink by $H / G$.

## 4.4 Weight Transfer

### 4.4.1 From MHA to GQA

Given MHA weights $W_Q^{(\text{MHA})} \in \mathbb{R}^{d \times H d_h}$ and 
$W_K^{(\text{MHA})}, W_V^{(\text{MHA})} \in \mathbb{R}^{d \times H d_h}$, we 
initialize GQA weights as:

$$
W_Q^{(\text{GQA})} = W_Q^{(\text{MHA})}
\tag{4.2}
$$

$$
W_K^{(\text{GQA})}_g = \text{mean}\!\left(W_K^{(\text{MHA})}_{h \in g}\right), \quad
W_V^{(\text{GQA})}_g = \text{mean}\!\left(W_V^{(\text{MHA})}_{h \in g}\right)
\tag{4.3}
$$

where the mean is over the heads in group $g$. This is a **uniform average** 
of the head weights within each group.

**Alternative:** We could pick the first head in each group or use learned 
pooling. Our experiments use averaging.

### 4.4.2 From GQA to GQA (Re-grouping)

If the source is already GQA with $G_\text{src}$ groups and we want 
$G_\text{tgt} < G_\text{src}$, we group the source KV heads:

$$
W_K^{(\text{tgt})}_g = \text{mean}\!\left(W_K^{(\text{src})}_{g' \in g}\right)
\tag{4.4}
$$

If $G_\text{tgt} = G_\text{src}$, the copy is exact.

### 4.4.3 Copy Procedure Pseudocode

**Algorithm 1:** `copy_weights` for GQA
```
Input:  src_attn (MHA or GQA)
        H, G, d_h
Output: GQA weights

1:  # Query weights are always copied verbatim
2:  W_Q ← extract_q(src_attn)
3:  copy(W_Q_gqa, W_Q)
4:
5:  # KV weights are grouped
6:  W_K_src ← extract_k(src_attn)   # shape (d, H * d_h) for MHA
7:  W_V_src ← extract_v(src_attn)
8:
9:  H_src ← W_K_src.shape[1] // d_h
10: r ← H_src // G
11:
12: for g in 0..G-1 do
13:     heads_in_g ← src heads [g*r, (g+1)*r)
14:     W_K_gqa[g] ← mean(W_K_src[heads_in_g])
15:     W_V_gqa[g] ← mean(W_V_src[heads_in_g])
16: end for
17:
18: # Output projection unchanged
19: W_O_gqa ← extract_o(src_attn)
20:
21: copy all projections into modules
22: freeze_base_weights()
```

## 4.5 Forward Pass

**Algorithm 2:** GQA forward
```
Input:  X ∈ R^{B × N × d}
        attention_mask
Output: Y ∈ R^{B × N × d}

1:  Q ← X W_Q            # (B, N, H, d_h)
2:  K ← X W_K            # (B, N, G, d_h)
3:  V ← X W_V            # (B, N, G, d_h)
4:
5:  if RoPE: (Q, K) ← apply_rotary(Q, K)
6:
7:  # Repeat KV heads to match Q heads
8:  r ← H // G
9:  K ← repeat_kv(K, r)     # (B, N, H, d_h)
10: V ← repeat_kv(V, r)
11:
12: # Standard scaled dot-product attention
13: scores ← Q K^T / √d_h
14: scores ← apply_causal_mask(scores)   # if decoder
15: scores ← apply_padding_mask(scores)
16: attn ← softmax(scores)
17: context ← attn V
18:
19: Y ← out_proj(context)
20: return Y
```

## 4.6 KV Cache Analysis

### 4.6.1 Per-Token Memory

For a single layer at sequence position $t$:

$$
\text{KV}^{(t)}_\text{MHA} = 2 H d_h \cdot \text{bytes}
\tag{4.5}
$$

$$
\text{KV}^{(t)}_\text{GQA} = 2 G d_h \cdot \text{bytes} = \frac{H}{G} \cdot \text{KV}^{(t)}_\text{MHA}
\tag{4.6}
$$

### 4.6.2 Total Cache

For sequence length $L$ and batch size $B$ across $N_l$ layers:

$$
\text{KV}_\text{total} = N_l \cdot L \cdot B \cdot 2 G d_h \cdot \text{bytes}
\tag{4.7}
$$

### 4.6.3 Concrete Numbers

For GPT-2 ($N_l = 12$, $H = 12$, $d_h = 64$, FP16):

| Config | $H_{KV}$ | KV/token (KB) | KV at L=1024 (MB) | KV at L=8192 (MB) |
|---|---|---|---|---|
| MHA | 12 | 36.9 | 37.7 | 302 |
| GQA (H/2) | 6 | 18.4 | 18.9 | 151 |
| GQA (H/4) | 3 | 9.2 | 9.4 | 75 |
| MQA | 1 | 3.1 | 3.1 | 25 |

## 4.7 Complexity Analysis

| Component | MHA | GQA |
|---|---|---|
| Q projection FLOPs | $\mathcal{O}(N d \cdot H d_h)$ | Same |
| K projection FLOPs | $\mathcal{O}(N d \cdot H d_h)$ | $\mathcal{O}(N d \cdot G d_h)$ |
| V projection FLOPs | Same as K | Same as K |
| Attention FLOPs | $\mathcal{O}(N^2 H d_h)$ | Same |
| Output projection FLOPs | Same | Same |
| **KV cache bytes** | $2 H d_h L B$ | $2 G d_h L B$ |

**Key insight.** GQA reduces KV cache and $K, V$ projection FLOPs proportionally 
to $G / H$, but **not** attention FLOPs or output projection FLOPs. The primary 
benefit is memory-bound inference.

## 4.8 Rank and Head Grouping

The number of KV heads $G$ must satisfy two constraints:

1. $G \le H$ (can't have more KV heads than query heads).
2. $H$ is divisible by $G$ (each group has $H / G$ heads).

If $H$ is prime, only $G = 1$ or $G = H$ are valid. For $H = 12$, valid values 
are $G \in \{1, 2, 3, 4, 6, 12\}$.

## 4.9 Theoretical Notes

### 4.9.1 MHA and MQA as Special Cases

At $G = H$, GQA reduces to MHA (each head has its own KV). At $G = 1$, GQA 
reduces to MQA (all heads share KV). GQA is a **continuous interpolation** 
between these extremes.

### 4.9.2 Expressiveness Bound

For a fixed sequence, the set of attention matrices expressible by GQA is a 
subset of MHA's expressible set. The reduction is bounded by the number of 
distinct KV heads:

$$
|\text{Expr}(\text{GQA})| \le \binom{H}{G}
\tag{4.8}
$$

relative to MHA's $|\text{Expr}(\text{MHA})|$ (informally, up to a 
factor depending on $d_h$).
