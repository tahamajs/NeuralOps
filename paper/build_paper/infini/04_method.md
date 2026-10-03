<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Infini-Attention replaces MHSA with a two-branch block:

1. **Local branch:** sliding-window attention over the input sequence.
2. **Memory branch:** attention to a learned compressive memory matrix.
3. **Fusion:** additive gated combination.

**Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                        │
    └────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌──────────┐             ┌──────────┐
    │ Q/K/V    │             │ Q        │
    │ proj     │             │          │
    └────┬─────┘             └────┬─────┘
         │                        │
         ▼                        ▼
    ┌──────────────┐      ┌──────────────┐
    │ Local        │      │ Memory       │
    │ Sliding-     │      │ attention    │
    │ window       │      │ on M         │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           └──────────┬──────────┘
                      ▼
              ┌───────────────┐
              │ out_local +   │
              │ σ(β)·out_mem  │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ out_proj      │
              └───────────────┘
```

## 4.2 Local Sliding-Window Branch

### 4.2.1 Mask Construction

For causal models, the local mask is:

$$
M_{ij}^\text{local} = \begin{cases} 0 & j \leq i \text{ and } i - j < W \\ -\infty & \text{otherwise} \end{cases}
\tag{4.1}
$$

For bidirectional models (encoders), the mask is symmetric:

$$
M_{ij}^\text{local} = \begin{cases} 0 & |i - j| < W \\ -\infty & \text{otherwise} \end{cases}
\tag{4.2}
$$

### 4.2.2 Local Attention

$$
\text{LocalAttn}_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_h}} + M^\text{local}\right) V
\tag{4.3}
$$

The softmax is computed in FP32 for stability, then cast back to the model 
dtype.

### 4.2.3 Why Sliding Window?

The window captures **local syntax**: adjacent words, subword fragments, 
short-range dependencies. Full attention over long sequences spends most 
of its compute on long-range connections that are often rare.

## 4.3 Compressive Memory Branch

### 4.3.1 The Memory Matrix

Each head has a learned memory:

$$
M_h \in \mathbb{R}^{d_c \times d_h}
\tag{4.4}
$$

where $d_c$ is the compression dimension. The full memory is 
$M \in \mathbb{R}^{H \times d_c \times d_h}$.

**Interpretation.** $M_h$ acts as $d_c$ learned "virtual key-value pairs". 
Since the same $M_h$ is used for every position, it captures global context 
that is shared across the sequence.

### 4.3.2 Memory Attention

$$
\text{MemAttn}_i = \text{softmax}\!\left(\frac{Q_i M^\top}{\sqrt{d_h}}\right) M
\tag{4.5}
$$

The softmax is over the $d_c$ memory slots. This has complexity 
$\mathcal{O}(L \cdot d_c \cdot d_h)$.

### 4.3.3 Memory Initialization

We initialize the memory with small random values:

$$
M \sim \mathcal{N}\!\left(0, \frac{1}{\sqrt{d_c}}\right) \cdot 0.001
\tag{4.6}
$$

The scale factor $0.001$ ensures the memory output is small initially, 
so the memory branch contributes minimally before training.

## 4.4 Gated Fusion

### 4.4.1 The Formula

$$
\text{output} = \text{LocalAttn} + \sigma(\beta) \cdot \text{MemAttn}
\tag{4.7}
$$

with $\beta \in \mathbb{R}$ a learnable scalar.

### 4.4.2 Why Additive?

Additive fusion preserves the pretrained output magnitude at initialization. 
Concatenation would change the output distribution. Multiplicative fusion 
would zero out the local branch when $\sigma(\beta) \approx 0$.

### 4.4.3 Why Sigmoid Gate?

The sigmoid $\sigma(\beta) \in (0, 1)$ bounds the memory contribution. This 
prevents the memory branch from dominating the output.

### 4.4.4 Initialization Strategy

We initialize $\beta = -5$. Then:

$$
\sigma(-5) \approx 0.0067
\tag{4.8}
$$

The memory contributes less than 1% at initialization. The model starts 
by exactly reproducing local sliding-window attention (which itself 
approximates the pretrained attention in the window).

**Why not initialize $\beta = 0$?** $\sigma(0) = 0.5$ would make the memory 
contribute 50% at start, potentially disrupting the pretrained behavior. 
$-\infty$ would freeze the memory. $-5$ is a good compromise.

## 4.5 Full Algorithm

**Algorithm 1:** Infini-Attention forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, window W
Output: Y ∈ R^{B×L×d}

1:  Q ← W_Q X + LoRA_Q(X)                ▷ Frozen W + LoRA
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:
7:  # Local branch
8:  S_local ← Q K^T / √d_h
9:  S_local ← S_local + causal_mask + window_mask
10: P_local ← softmax(S_local)
11: out_local ← P_local V
12:
13: # Memory branch
14: S_mem ← Q M^T / √d_h
15: P_mem ← softmax(S_mem)
16: out_mem ← P_mem M
17:
18: # Gated fusion
19: gate ← sigmoid(β)
20: combined ← out_local + gate · out_mem
21:
22: combined ← concat_heads(combined)
23: Y ← W_O combined + LoRA_O(combined)
24: return Y
```

## 4.6 Weight Transfer

Infini-Attention transfers weights from the pretrained attention via the 
same routine as NHA/EAT (§5).

## 4.7 Complexity Analysis

| Component | Time | Space |
|---|---|---|
| Q/K/V projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Local attention | $\mathcal{O}(L W d_h H)$ | $\mathcal{O}(L W H)$ |
| Memory attention | $\mathcal{O}(L d_c d_h H)$ | $\mathcal{O}(L d_c H)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L W d + L d_c d)$ | $\mathcal{O}(L W + L d_c)$ |

**Comparison with full attention:**

| Method | Time | Space |
|---|---|---|
| Full MHSA | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |
| Sliding window | $\mathcal{O}(L W d)$ | $\mathcal{O}(L W)$ |
| **Infini-Attention** | $\mathcal{O}(L(W + d_c) d)$ | $\mathcal{O}(L(W + d_c))$ |

For $W = d_c = 128$ and $L = 4096$:

| Method | Relative time |
|---|---|
| Full MHSA | 1.00× |
| Sliding window | 0.031× |
| **Infini-Attention** | **0.062×** |

The memory roughly doubles the cost of pure sliding window, but is still 
~16× cheaper than full attention.

## 4.8 Parameter Analysis

Infini-Attention **adds** parameters to the model. Per attention layer:

| Component | Parameters |
|---|---|
| Memory $M$ | $H \cdot d_c \cdot d_h$ |
| Gate $\beta$ | 1 |
| LoRA Q | $r(d + H d_h)$ |
| LoRA K | $r(d + H_{kv} d_h)$ |
| LoRA V | $r(d + H_{kv} d_h)$ |
| LoRA O | $r(H d_h + d)$ |
| **Total** | see below |

For DistilBERT ($H = 12$, $d_h = 64$, $d_c = 128$, $d = 768$, $r = 16$, 
$N_\ell = 6$):

| Component | Per layer | 6 layers |
|---|---|---|
| Memory | 98,304 | 589,824 |
| Gate | 1 | 6 |
| LoRA (all 4) | 4 × 24,576 = 98,304 | 589,824 |
| **Total** | **196,609** | **1,179,654** |

This is a **+1.76%** parameter increase on DistilBERT.

## 4.9 Theoretical Notes

### 4.9.1 Effect of Gate Initialization

**Claim.** With $\beta = -5$, the memory contributes less than 1% to the 
output at initialization.

**Proof.** $\sigma(-5) = \frac{1}{1 + e^5} \approx 0.0067$. Since 
$\|\text{MemAttn}\|_2$ is bounded by $\|M\|_F$, the memory output is at 
most $\sigma(-5) \|M\|_F \approx 0.007 \|M\|_F$. For small $\|M\|_F$ 
(initialized at $0.001$ scale), the contribution is negligible. $\square$

### 4.9.2 Memory Complexity

**Claim.** The memory matrix $M$ has complexity $\mathcal{O}(H d_c d_h)$.

**Proof.** Per head, $M_h \in \mathbb{R}^{d_c \times d_h}$ has $d_c d_h$ 
parameters. There are $H$ heads, so total is $H d_c d_h$. $\square$

### 4.9.3 Fusion Preserves Pretrained Behavior

**Claim.** If the local branch reproduces the pretrained attention output 
within the window and $\beta \to -\infty$, then Infini-Attention reduces to 
sliding-window attention.

**Proof.** As $\beta \to -\infty$, $\sigma(\beta) \to 0$, so 
$\text{output} \to \text{LocalAttn}$. $\square$

---