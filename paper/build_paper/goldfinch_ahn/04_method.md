<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

GoldFinch-AHN has the following structure:

```
                    Input X
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
    ┌──────┐       ┌──────┐      ┌───────────┐
    │  Q   │       │  K   │      │  Linear   │
    │ proj │       │ proj │      │  (delta)  │
    └──┬───┘       └──┬───┘      └─────┬─────┘
       │              │                │
       ▼              ▼                ▼
   ┌──────────────────────┐      ┌───────────┐
   │ Sliding-window       │      │ long_out  │
   │ attention            │      └─────┬─────┘
   └────────┬─────────────┘            │
            │ short_out                │
            │                          │
       ┌────┴──────────────────────────┴────┐
       │                                    │
       │  gate(x) → per-token mixing        │
       │  fused = g * short + (1-g) * long  │
       └────────┬───────────────────────────┘
                │
                ▼
       ┌──────────────────┐
       │ out_proj         │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Preservation     │
       │ gate: α·out +    │
       │ (1-α)·x          │
       └────────┬─────────┘
                │
                ▼
              Output
```

## 4.2 Sliding-Window Attention

We compute standard scaled dot-product attention with a band mask:

$$
\text{scores}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
\tag{4.1}
$$

Apply window mask:

$$
M_{ij} = \begin{cases} 1 & |i - j| \leq W \\ 0 & \text{otherwise} \end{cases}
\tag{4.2}
$$

For causal decoders:

$$
M_{ij} = \begin{cases} 1 & 0 \leq i - j \leq W \\ 0 & \text{otherwise} \end{cases}
\tag{4.3}
$$

Then:

$$
\text{short}_i = \sum_j \text{softmax}_j(\text{scores}_{ij} + \log M_{ij}) v_j
\tag{4.4}
$$

## 4.3 Long-Memory Branch (Linear Projection)

**Important.** The code refers to this as `GatedDeltaNetLayer` but 
implements:

```python
long_out = self.delta_mem(x)  # = Linear(x)
```

There is no recurrence, no gating, no state. It is a single linear 
projection $W_m \in \mathbb{R}^{d \times d}$. Its role in the architecture 
is:

$$
\text{long}_t = W_m x_t + b_m
\tag{4.5}
$$

**Interpretation.** The long-memory branch is a per-token learned 
transformation. It cannot capture long-range dependencies because it 
processes each token independently.

## 4.4 Input-Dependent Gate

The gate:

$$
g_t = \sigma(W_g x_t + b_g) \in \mathbb{R}^H
\tag{4.6}
$$

with $W_g \in \mathbb{R}^{d \times H}$ and $b_g \in \mathbb{R}^H$.

**Mean over heads:** In forward, the gate is reduced to a scalar per token:

$$
g'_t = \frac{1}{H} \sum_{h=1}^H g_{t,h}
\tag{4.7}
$$

**Fusion:**

$$
\text{fused}_t = g'_t \cdot \text{short}_t + (1 - g'_t) \cdot \text{long}_t
\tag{4.8}
$$

## 4.5 Output Projection and Preservation Gate

Output projection:

$$
\text{out} = W_O \cdot \text{fused} + b_O
\tag{4.9}
$$

Then the preservation gate:

$$
\alpha = \sigma(w_\alpha) \approx 0.88 \text{ at init}
\tag{4.10}
$$

$$
Y = \alpha \cdot \text{out} + (1 - \alpha) \cdot x
\tag{4.11}
$$

**Interpretation.** The preservation gate blends the block output with the 
input. At $\alpha = 0.88$, the block contributes 88% of the variance and 
the input contributes 12%. This keeps the model close to its pretrained 
behavior.

## 4.6 Parameter Analysis

| Component | Baseline | GoldFinch |
|---|---|---|
| Q/K/V/O projections | $4 d^2$ | $4 d^2 + \text{LoRA}$ |
| Gate $W_g$ | 0 | $d H + H$ |
| Long-memory $W_m$ | 0 | $d^2 + d$ |
| Preservation $\alpha$ | 0 | 1 |
| Output scale | 0 | 1 |

**Example (DistilBERT, $d = 768$, $H = 12$):**
- Gate: $768 \times 12 + 12 = 9{,}228$
- Long-memory: $768^2 + 768 = 590{,}592$
- LoRA per projection: $2 \times 16 \times 768 = 24{,}576$ × 4 = 98,304
- **Total per layer: ~698,124**

**Δ% per layer (DistilBERT): ~29.6% of a layer's params.**
**Δ% total (DistilBERT): ~5-6%.**

## 4.7 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| Sliding window | $\mathcal{O}(LWd)$ | $\mathcal{O}(LW)$ |
| Long-memory (Linear) | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| Gate | $\mathcal{O}(LdH)$ | $\mathcal{O}(LH)$ |
| Output proj | $\mathcal{O}(Ld^2)$ | $\mathcal{O}(Ld)$ |
| **Total** | $\mathcal{O}(Ld^2 + LWd)$ | $\mathcal{O}(LW)$ |
| **Baseline MHSA** | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2)$ |

For $W \ll L$: GoldFinch is $\mathcal{O}(L d^2)$ — linear in $L$.

## 4.8 Algorithm

**Algorithm 1:** GoldFinch-AHN forward
```
Input:  X ∈ R^{B×L×d}, is_decoder, W (window)
Output: Y ∈ R^{B×L×d}

1: Q ← X W_Q;  K ← X W_K;  V ← X W_V
2: if RoPE: (Q, K) ← Rotary(Q, K)
3: if GQA: (K, V) ← repeat_kv(K, V)
4:
5: # Sliding-window attention
6: scores ← Q K^T / sqrt(d)
7: M ← band_mask(L, W, causal=is_decoder)
8: scores ← scores + log(M)
9: attn ← softmax(scores)
10: short ← attn @ V
11:
12: # Long-memory branch (LINEAR PROJECTION)
13: long ← Linear_m(X)
14:
15: # Input-dependent gate (mean over heads)
16: gate ← mean(sigmoid(W_g X + b_g))
17: fused ← gate * short + (1 - gate) * long
18:
19: out ← out_proj(fused)
20: α ← sigmoid(output_gate)   ▷ init ≈ 0.88
21: Y ← α * out + (1 - α) * X
22: return Y
```

## 4.9 Theoretical Notes

### 4.9.1 Sliding-Window Complexity

For fixed $W$ and sequence length $L$:

$$
T_\text{window} = \mathcal{O}(L W d)
\tag{4.12}
$$

This is **linear** in $L$, matching linear attention complexity.

### 4.9.2 The Long-Memory Branch Contributes Almost Nothing

Since the long-memory branch is $W_m x_t$ (per-token), it cannot encode 
sequence-level information. Its only role is to add a per-token 
transformation that the gate can blend in.

**Effect.** With gate $g \approx 0.86$, the long-memory contribution is 
$\sim 14\%$ of fused. Given that $W_m$ is randomly initialized and only 
LoRA-trained, this contribution is small.

### 4.9.3 Preservation Gate at Initialization

At initialization:
- $\alpha = \sigma(2.0) \approx 0.88$
- $Y = 0.88 \cdot \text{out} + 0.12 \cdot x$

Since `out` depends on randomly-initialized $W_m$ (but not on $W_Q, W_K, W_V$ 
which are pretrained), the output has a mix of pretrained + random.

**LoRA recovery** must align the gate + long-memory + output scale to 
produce useful output.

---