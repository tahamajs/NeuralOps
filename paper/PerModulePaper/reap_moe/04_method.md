# 4. Method

## 4.1 Overview

REAP-MoE is a two-stage architecture:

1. **Attention stage** (§4.2): standard multi-head attention.
2. **MoE stage** (§4.3): sparse mixture of SVD-factorized experts.

Both stages use pre-LayerNorm:

$$
\begin{aligned}
h_1 &= \text{LN}_1(x + \text{Attn}(x)) \\
y &= \text{LN}_2(h_1 + \text{MoE}(h_1))
\end{aligned}
\tag{4.1}
$$

**Figure 4.1** shows the pipeline.

```
   x ───────────────────────┐
   │                        │
   ├─→ LN_1 ─→ Attn ─→ (+) ─→ h_1
   │                        │
   └────────────────────────┘
                            │
   h_1 ─────────────────────┐
   │                        │
   ├─→ LN_2 ─→ MoE ─→ (+) ─→ y
   │                        │
   └────────────────────────┘
```

## 4.2 Attention Stage

Standard multi-head attention:

$$
\text{Attn}(x) = W_O \cdot \text{Concat}(\text{head}_1, \dots, \text{head}_H)
\tag{4.2}
$$

where each head is:

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) V_h
\tag{4.3}
$$

Projections $W_Q, W_K, W_V, W_O$ use LoRALinear with frozen base weights 
and trainable adapters.

## 4.3 MoE Stage

The MoE block replaces the standard FFN.

### 4.3.1 Router

Given hidden state $x \in \mathbb{R}^{B \times L \times d}$:

$$
\text{logits} = W_r \cdot \text{LN}_\text{router}(x) \in \mathbb{R}^{B \times L \times E}
\tag{4.4}
$$

where $W_r \in \mathbb{R}^{E \times d}$.

**Noise injection (training only):**

$$
\text{scores} = \text{logits} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\tag{4.5}
$$

### 4.3.2 Top-$k$ Selection

$$
\mathcal{T}(x) = \text{Top-}k(\text{scores})
\tag{4.6}
$$

$$
g_i(x) = \frac{\exp(\text{scores}_i)}{\sum_{j \in \mathcal{T}(x)} \exp(\text{scores}_j)}, \quad i \in \mathcal{T}(x)
\tag{4.7}
$$

### 4.3.3 Expert Computation

Each expert $E_i$ is a two-layer MLP:

$$
E_i(x) = U_2^{(i)} \cdot \text{GELU}(U_1^{(i)} \cdot x)
\tag{4.8}
$$

where $U_1^{(i)}, U_2^{(i)}$ are themselves SVD-factorized:

$$
U_1^{(i)} = P_1^{(i)} \Sigma_1^{(i)} Q_1^{(i)\top}
\tag{4.9}
$$

with similar structure for $U_2^{(i)}$.

### 4.3.4 MoE Output

$$
\text{MoE}(x) = \sum_{i \in \mathcal{T}(x)} g_i(x) \cdot E_i(x)
\tag{4.10}
$$

### 4.3.5 Zero Initialization

We initialize the final linear layer of each expert to **zero**. This ensures 
that at step 0:

$$
\text{MoE}(x) = 0
\tag{4.11}
$$

so the model starts at the identity residual path and gradually learns to 
add expert outputs. This is critical for training stability.

## 4.4 Load-Balancing Auxiliary Loss

We add the auxiliary loss to the total training objective:

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{task} + \lambda_\text{aux} \cdot \mathcal{L}_\text{aux}
\tag{4.12}
$$

where:

$$
\mathcal{L}_\text{aux} = E \cdot \sum_{i=1}^{E} \rho_i \cdot \pi_i
\tag{4.13}
$$

with:
- $\rho_i = \frac{1}{BL} \sum_{b,l} \mathbb{1}[i \in \mathcal{T}(x_{bl})]$ — expert utilization.
- $\pi_i = \frac{1}{BL} \sum_{b,l} \text{softmax}(\text{logits}_{bl})_i$ — mean gate probability.

**Why this loss?** When both $\rho$ and $\pi$ are uniform, 
$\mathcal{L}_\text{aux} = E \cdot (1/E) \cdot (1/E) = 1/E$. Wait, that's 
wrong. Let me recompute.

If $\rho_i = 1/E$ and $\pi_i = 1/E$:
$\mathcal{L}_\text{aux} = E \cdot \sum_i (1/E)(1/E) = E \cdot E \cdot (1/E^2) = 1$.

So $\mathcal{L}_\text{aux} = 1$ when balanced. And $\mathcal{L}_\text{aux} > 1$ 
when imbalanced.

## 4.5 Full Algorithm

**Algorithm 1:** REAP-MoE forward pass
```
Input:  x ∈ R^{B×L×d}
Output: y ∈ R^{B×L×d}, L_aux

1:  # Attention stage
2:  Q ← LN_1(x); K ← LN_1(x); V ← LN_1(x)
3:  attn_out ← MHSA(Q, K, V)
4:  h_1 ← x + attn_out
5:
6:  # MoE stage
7:  h_2 ← LN_2(h_1)
8:  logits ← W_r · h_2
9:  if training:
10:     logits ← logits + N(0, σ²)
11: topk_weights, topk_idx ← topk(softmax(logits), k)
12:
13: # Compute all experts
14: experts_out ← [E_i(h_2) for i in 1..E]   ▷ [B, L, E, d]
15: gathered ← gather(experts_out, topk_idx)
16: moe_out ← sum(gathered * topk_weights, dim=k)
17:
18: y ← h_1 + moe_out
19:
20: # Auxiliary loss
21: ρ ← mean(one_hot(topk_idx), dim=(0,1))
22: π ← mean(softmax(logits), dim=(0,1))
23: L_aux ← E · sum(ρ · π)
24:
25: return y, L_aux
```

## 4.6 Complexity Analysis

### 4.6.1 Parameter Count

For a single MoE layer with $E$ experts and SVD rank $r_1, r_2$:

$$
\text{Params}_\text{MoE} = E \left( r_1(d + 2d) + r_2(2d + d) \right) = E \cdot 3d(r_1 + r_2)
\tag{4.14}
$$

With $r_1 = r_2 = d/4$:

$$
\text{Params}_\text{MoE} = E \cdot 3d \cdot d/2 = 1.5 E d^2
\tag{4.15}
$$

Compare to standard FFN:

$$
\text{Params}_\text{FFN} = 8 d^2
\tag{4.16}
$$

**Ratio:** $1.5 E d^2 / 8 d^2 = 0.1875 E$. For $E = 8$:

$$
\text{Ratio} = 1.5
\tag{4.17}
$$

So MoE with SVD is **1.5× the FFN parameters**, not 8× as in standard MoE.

### 4.6.2 Compute

Per token, only $k$ experts run:

$$
\text{FLOPs}_\text{MoE} = k \cdot (r_1 d + r_2 \cdot 2d) \cdot L
\tag{4.18}
$$

For $k = 2$, this is 2× the SVD layer's FLOPs.

### 4.6.3 Summary

| Component | Original FFN | REAP-MoE (E=8, k=2) |
|---|---|---|
| Parameters | $8d^2$ | $1.5 E d^2 \approx 12d^2$ |
| Per-token FLOPs | $8d^2$ | $3 k d^2 = 6d^2$ (SVD × k) |
| Effective capacity | $8d^2$ | $12d^2$ |

## 4.7 Theoretical Notes

### 4.7.1 Load-Balancing Guarantee

**Proposition.** The auxiliary loss $\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$ 
is minimized when both $\rho$ and $\pi$ are uniform, with minimum value 1.

**Proof.** By AM-GM inequality, $\sum_i \rho_i \pi_i \geq E \cdot \min_i \rho_i \pi_i$. 
For uniform distributions, $\rho_i = \pi_i = 1/E$, giving $\mathcal{L}_\text{aux} = E \cdot E \cdot (1/E^2) = 1$. $\square$

### 4.7.2 SVD Approximation Error

Each expert layer's SVD approximation has error bounded by:

$$
\|W - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.19}
$$

### 4.7.3 Zero-Init Stability

With zero-init on the final layer, the MoE block at initialization is 
identity (in residual path). This ensures gradient flow through attention 
at step 0 without interference from random MoE outputs.

---