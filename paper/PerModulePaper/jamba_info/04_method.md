# 4. Method

## 4.1 Overview

JambaInfo is a two-branch hybrid block:

- **Branch 1: Differential attention** (noise-reduced).
- **Branch 2: Concept-bottlenecked SSM** (IB-motivated compression).

The branches are fused by a learned scalar gate.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
       ┌──────┴──────┐
       ▼             ▼
   ┌───────┐    ┌───────────────┐
   │ Diff  │    │  Bottleneck   │
   │ Attn  │    │  W_down →     │
   │       │    │  GELU →       │
   │       │    │  W_up         │
   └───┬───┘    └───────┬───────┘
       │                ▼
       │            ┌───────┐
       │            │  SSM  │
       │            └───┬───┘
       │ α=σ(w)         │
       │  ┌─────────────┘
       ▼  ▼
   (1-α)⊙A + α⊙M
       │
       ▼
     out_proj
       │
       ▼
    Output Y
```

## 4.2 Differential Attention Branch

Same as HymbaTransX:

$$
Q_1 = X W_{Q_1}, \quad K_1 = X W_{K_1}, \quad
Q_2 = X W_{Q_2}, \quad K_2 = X W_{K_2}
\tag{4.1}
$$

$$
A_1 = \text{softmax}(Q_1 K_1^\top / \sqrt{d_h}), \quad
A_2 = \text{softmax}(Q_2 K_2^\top / \sqrt{d_h})
\tag{4.2}
$$

$$
A_\text{diff} = \text{ReLU}(A_1 - 0.2 A_2), \quad
\tilde{A}_\text{diff} = \frac{A_\text{diff}}{\sum A_\text{diff} + \epsilon}
\tag{4.3}
$$

$$
A_\text{out} = \tilde{A}_\text{diff} V, \quad V = X W_V
\tag{4.4}
$$

## 4.3 Concept-Bottlenecked SSM Branch

**Step 1 — Bottleneck projection:**

$$
z = W_\text{up} \cdot \text{GELU}(W_\text{down} X)
\tag{4.5}
$$

with $W_\text{down} \in \mathbb{R}^{b \times d}$, $W_\text{up} \in \mathbb{R}^{d \times b}$.

**Step 2 — SSM on concept representation:**

For decoders:
$$
M_\text{concept} = \text{SSM}(z)
\tag{4.6}
$$

For encoders (bidirectional):
$$
M_\text{fwd} = \text{SSM}(z), \quad M_\text{bwd} = \text{flip}(\text{SSM}(\text{flip}(z)))
\tag{4.7}
$$

$$
M_\text{concept} = 0.5(M_\text{fwd} + M_\text{bwd})
\tag{4.8}
$$

## 4.4 Why a Concept Bottleneck?

### 4.4.1 Information-Theoretic Motivation

The SSM state $h_t \in \mathbb{R}^d$ has bounded capacity. Feeding it a 
$d$-dimensional input with high redundancy wastes capacity on noise. The 
bottleneck reduces the input to $b = 256$ dimensions of "concepts", 
concentrating information.

### 4.4.2 Parameter Reduction

Without bottleneck: SSM has $d$-dimensional input and output.
With bottleneck: SSM still $d \times d$, but the input to SSM has fewer 
unique directions.

### 4.4.3 Empirical Observation

**Figure 4.2** (deferred to Section 8) shows that the SSM's recurrent 
state has effective rank closer to $b$ than $d$ — confirming the 
bottleneck dimension is appropriate.

## 4.5 Fusion

$$
\alpha = \sigma(w), \quad w_0 = -2
\tag{4.9}
$$

$$
\text{combined} = (1 - \alpha) \odot A_\text{out} + \alpha \odot M_\text{concept}
\tag{4.10}
$$

**Note the swap.** HymbaTransX uses $\alpha A + (1-\alpha) M$; we use 
$(1-\alpha) A + \alpha M$. Combined with $\alpha_0 = 0.12$, this favors 
attention initially, letting the model learn how much SSM it needs.

## 4.6 Dual Operational Modes

JambaInfo supports two modes:

| Mode | Description | α init | Use case |
|---|---|---|---|
| `retrieval` | Default. Bias toward attention. | -2 | Classification, QA |
| `compression` | Bias toward SSM. | +2 | Long-context, generation |

The `mode` parameter is passed at construction; the difference is the 
initial value of $w$ and a slight rescaling of the SSM output.

## 4.7 Sparsity Parameter

The `sparsity` parameter (default 0.5) controls an optional sparsity 
penalty on the differential attention. When active, it encourages the 
$A_\text{diff}$ map to be sparse (few nonzero entries).

In the current implementation, `sparsity` is a placeholder — future work 
will add the penalty.

## 4.8 Algorithm

**Algorithm 1:** JambaInfo forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
        position_embeddings
Output: Y ∈ R^{B×L×d}

 1: Q1, K1 ← W_q1(X), W_k1(X)
 2: Q2, K2 ← W_q2(X), W_k2(X)
 3: V ← W_v(X)
 4: if pos_emb: apply RoPE
 5: if GQA: repeat_kv
 6: A1 ← softmax(Q1 K1^T / √d_h)
 7: A2 ← softmax(Q2 K2^T / √d_h)
 8: A_diff ← ReLU(A1 - 0.2 A2)
 9: A_out ← normalize(A_diff) @ V
10: z ← W_up(GELU(W_down(X)))       ▷ Concept
11: if decoder: M ← SSM(z)
12: else: M ← 0.5(SSM(z) + flip(SSM(flip(z))))
13: α ← sigmoid(w)
14: out ← (1-α) A_out + α M
15: Y ← out_proj(out)
16: return Y
```

## 4.9 Complexity Analysis

Per block:

| Component | Time | Params |
|---|---|---|
| Q/K/V | $O(L d^2)$ | $3 d^2$ |
| W_q2, W_k2 | $O(L d^2)$ | $2 d^2$ |
| Diff attention | $O(L^2 d)$ | 0 |
| Bottleneck | $O(L d b)$ | $2 b d$ |
| SSM | $O(L d)$ | $2 d^2$ |
| out_proj | $O(L d^2)$ | $d^2$ |
| **Total** | $O(L^2 d)$ | $8 d^2 + 2bd$ |

Compared to baseline ($4 d^2$), we have ~2× more parameters.

## 4.10 Parameter Analysis

For $d = 768$, $b = 256$:

| Component | Baseline | JambaInfo |
|---|---|---|
| Q, K, V, O | 4×589,824 = 2.36M | — |
| W_q1, W_k1 | — | 1.18M |
| W_q2, W_k2 | — | 1.18M |
| W_v | — | 0.59M |
| Bottleneck | — | 0.39M |
| SSM | — | 0.59M |
| W_o | — | 0.59M |
| **Total** | **2.36M** | **4.52M** |
| **Increase** | — | **+92%** |

## 4.11 Training

Trainable:
- LoRA adapters on all attention projections.
- $W_{Q_2}, W_{K_2}$ (Kaiming init).
- Bottleneck $W_\text{down}, W_\text{up}$.
- SSM parameters.
- $\alpha$ (scalar).

Frozen:
- Base Q/K/V/O weights.

**Trainable count:** ~0.8M per layer.

---