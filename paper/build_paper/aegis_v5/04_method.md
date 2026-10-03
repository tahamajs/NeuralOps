# 4. Method

## 4.1 Overview

AEGIS-V5 has three main stages:

1. **SVD-factorized projections** (§4.2): each $Q/K/V/O$ projection is 
   decomposed with energy-based rank selection and frozen.
2. **Split-head attention** (§4.3): heads partitioned into local and global.
3. **Adaptive gating** (§4.4): per-head gate fuses local and global.

```
        Input X
           │
           ▼
    ┌──────────────────┐
    │ SVD Q/K/V/O      │  (frozen, energy-ranked)
    │ + LoRA adapters  │  (trainable)
    └────────┬─────────┘
             │
    ┌────────┴─────────┐
    ▼                  ▼
 ┌──────────┐    ┌───────────┐
 │ Local    │    │ Global    │
 │ Window   │    │ RFF       │
 │ (2H/3)   │    │ (H/3)     │
 └────┬─────┘    └─────┬─────┘
      │                │
      └───────┬────────┘
              ▼
      ┌───────────────┐
      │ Per-head      │
      │ Gating σ(g)   │
      └───────┬───────┘
              │
              ▼
         Output
```

## 4.2 SVD-Factorized Projections

### 4.2.1 Decomposition

Given a weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$:

1. Compute the SVD:
   $$
   W = U \Sigma V^\top
   \tag{4.1}
   
<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$
2. Select energy-based rank:
   $$
   r^* = \min \left\{ r : \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2} \geq \eta \right\}
   \tag{4.2}
   
<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$
3. Factor into $W \approx U_r V_r$ where:
   $$
   U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
   V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
   \tag{4.3}
   
<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

### 4.2.2 Parameter Reduction

| Configuration | Parameters |
|---|---|
| Original | $d_\text{out} \cdot d_\text{in}$ |
| SVD at rank $r$ | $r(d_\text{out} + d_\text{in})$ |

For $d = 768$ and $r = 384$ (energy 0.95):

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| SVD r=384 | 589,824 | **1.00×** |

**Critical observation.** At $r = 384 = d/2$, SVD parameters equal original. 
For meaningful compression, we need $r < d/2$.

### 4.2.3 LoRA on Frozen Factors

We freeze $(U_r, V_r)$ and add LoRA adapters:

$$
y = U_r V_r x + \frac{\alpha}{r_L} B A x
\tag{4.4}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

Only $A, B$ are trained.

**Note.** AEGIS-V5 uses a `FactorisedLinear` that applies SVD as a **stacked** 
composition (`down` then `up`), not a reconstruction. This is critical for 
correctness.

## 4.3 Split-Head Attention

### 4.3.1 Head Partition

We split $H$ heads:

$$
H_\text{local} = \lceil 2H/3 \rceil, \quad
H_\text{global} = H - H_\text{local}
\tag{4.5}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

For $H = 12$:
- $H_\text{local} = 8$
- $H_\text{global} = 4$

### 4.3.2 Local Branch

Heads $0 \ldots H_\text{local} - 1$ use sliding window attention:

$$
\text{Out}_\text{local}[i] = \text{softpick}\!\left(\frac{q_i K_\text{local}^\top}{\sqrt{d_h}}\right) \cdot V_\text{local}
\tag{4.6}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

with a window mask $|i - j| \leq W/2$.

### 4.3.3 Global Branch

Heads $H_\text{local} \ldots H - 1$ use kernel linear attention:

$$
\text{Out}_\text{global}[i] = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{4.7}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

with RFF feature map $\phi(x) = \exp(W_\text{rff} x) / \sqrt{F}$.

### 4.3.4 Concatenation

The outputs are concatenated:

$$
\text{Out} = \text{Concat}(\text{Out}_\text{local}, \text{Out}_\text{global}) \in \mathbb{R}^{B \times H \times L \times d_h}
\tag{4.8}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

## 4.4 Adaptive Gating

### 4.4.1 Per-Head Gate

Each head has a learnable scalar $g_h$:

$$
\text{Out}_\text{fused}[i] = \sigma(g_h + 2.0) \cdot \text{Out}_\text{local}[i] + (1 - \sigma(g_h + 2.0)) \cdot \text{Out}_\text{global}[i]
\tag{4.9}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

where $g_h$ is initialized to 0 (so $\sigma(2.0) \approx 0.88$).

### 4.4.2 Why Bias 2.0?

At initialization, we want the model to prioritize local attention (known 
to be safe). As training progresses, the gate can shift toward global.

## 4.5 Decoder Fallback

For decoder models, the local+global split **does not preserve causality**:

- Local branch: causal window is fine.
- Global branch: prefix scan preserves causality.

**But:** For very short sequences ($N < W$), the local branch covers the 
whole sequence, making the split redundant. We fall back to standard dense 
causal attention:

$$
\text{if } \text{is\_decoder} : \text{standard causal softmax}
\tag{4.10}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

This preserves correctness at the cost of quadratic complexity for decoders.

## 4.6 Full Algorithm

**Algorithm 1:** AEGIS-V5 forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, is_decoder
Output: Y ∈ R^{B×L×d}

1:  Q ← SVD_Q(X); K ← SVD_K(X); V ← SVD_V(X)
2:  if RoPE: Q, K ← RoPE(Q, K)
3:  if GQA: K, V ← repeat_kv(K, V)
4:  if is_decoder:
5:      # Fallback to standard causal attention
6:      scores ← Q K^T / √d_h
7:      scores ← mask_causal(scores)
8:      out ← softmax(scores) @ V
9:  else:
10:     # Split-head hybrid attention
11:     Q_local ← Q[:, :H_local]; K_local ← K[:, :H_local]; V_local ← V[:, :H_local]
12:     Q_global ← Q[:, H_local:]; K_global ← K[:, H_local:]; V_global ← V[:, H_local:]
13:     Out_local ← local_window(Q_local, K_local, V_local, W)
14:     Out_global ← kernel_linear(Q_global, K_global, V_global, φ)
15:     # Per-head gate
16:     g ← σ(blend_logits + 2.0)
17:     Out ← g · Out_local + (1 - g) · Out_global
18: Y ← SVD_O(Out)
19: return Y
```

## 4.7 Complexity Analysis

### 4.7.1 Encoder

| Component | Complexity |
|---|---|
| SVD projections | $\mathcal{O}(L r (d + d_h))$ |
| Local attention | $\mathcal{O}(L W d_h \cdot H_\text{local})$ |
| Kernel attention | $\mathcal{O}(L F d_h \cdot H_\text{global})$ |
| Output projection | $\mathcal{O}(L r d)$ |
| **Total** | $\mathcal{O}(L(W H_\text{local} + F H_\text{global}) d_h)$ |

For fixed $W, F, H_\text{local}, H_\text{global}$: linear in $L$.

### 4.7.2 Decoder

Quadratic fallback: $\mathcal{O}(L^2 d_h)$.

## 4.8 Parameter Analysis

| Component | Baseline | AEGIS-V5 | Change |
|---|---|---|---|
| Q/K/V/O (SVD) | $4 d^2$ | $4 r(d + d_h)$ | depends on $r$ |
| LoRA adapters | 0 | $4 r_L (d + d_h)$ | +small |
| RFF proj | 0 | $H_\text{global} \cdot d_h \cdot F$ | +medium |
| blend_logits | 0 | $H$ | +negligible |

**Critical.** Whether AEGIS-V5 reduces parameters depends entirely on $r$.

For $d = 768$, $r = 384$ (energy 0.95 default):
- SVD: no compression (ratio 1.00×)
- LoRA: +98,304
- RFF: $4 \cdot 64 \cdot 128 = 32,768$
- **Net: +131,072 params per layer**

For $d = 768$, $r = 128$:
- SVD: $128 \cdot 1536 = 196,608$ (vs 589,824) → **3× compression**
- LoRA: +98,304
- RFF: +32,768
- **Net: -262,144 params per layer**

**We are explicit:** The parameter effect depends on the SVD rank.

## 4.9 Theoretical Notes

### 4.9.1 SVD Approximation Error

If $W = U\Sigma V^\top$, then:

$$
\|W - U_r \Sigma_r V_r^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.11}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

The energy-based rank ensures this is below $\sqrt{1 - \eta} \|W\|_F$.

### 4.9.2 Kernel Approximation

Standard RFF bound (Rahimi & Recht, 2007):

$$
\sup_{x, y} |\phi(x)^\top \phi(y) - e^{x \cdot y}| \leq \epsilon
\tag{4.12}

<!-- source: neuralops/attentions/aegis_v5.py line 236 -->
$$

with $F = \mathcal{O}(d_h \log(1/\epsilon^2))$.

### 4.9.3 Causality

**Claim.** For decoders, the fallback preserves causality.

**Proof.** Fallback uses standard causal softmax with triangular mask. $\square$

**Claim.** For encoders, the split-head design preserves non-causality 
(no masking needed).

**Proof.** Local window covers $|i - j| \leq W/2$; kernel attention covers 
all $j$. Both are bidirectional by design. $\square$

---