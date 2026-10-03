<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

HALO-ReGLA runs two attention streams in parallel and combines them with 
per-head gates:

1. **Dense stream** $O^\text{dense}$: standard softmax attention.
2. **Linear stream** $O^\text{lin}$: rectified kernel attention.
3. **Gate** $g$: per-head per-token sigmoid.
4. **Fusion** $O = g \odot O^\text{dense} + (1 - g) \odot O^\text{lin}$.

**Figure 4.1** illustrates the pipeline.

```
        Q, K, V ∈ R^{B × H × L × d_h}
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │ Dense   │      │ Linear   │
   │ softmax │      │ (ELU+1)  │
   │   O_d   │      │   O_l    │
   └────┬────┘      └────┬─────┘
        │                │
        └────────┬───────┘
                 ▼
        ┌─────────────────┐
        │  Gate g(x)      │
        │  g·O_d+(1-g)·O_l│
        └────────┬────────┘
                 ▼
              out_proj
```

## 4.2 Dense Stream

Standard softmax attention with causal masking and padding:

$$
S_{ij} = \frac{Q_i^\top K_j}{\sqrt{d_h}}, \quad
S_{ij} \leftarrow S_{ij} + \text{mask}_{ij}, \quad
O^\text{dense}_i = \sum_j \text{softmax}(S_i)_j V_j
\tag{4.1}

<!-- source: neuralops/attentions/halo_regla.py line 447 -->

<!-- source: neuralops/attentions/halo_regla.py line 447 -->
$$

**Complexity.** $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory.

**Masks applied.**
- Causal: $S_{ij} = -\infty$ for $j > i$.
- Padding: $S_{ij} = -\infty$ for invalid keys $j$.

## 4.3 Linear Stream

Rectified kernel attention with cumulative sums for causality:

### 4.3.1 Feature Map

$$
\phi(x) = \mathrm{elu}(x) + 1 \in \mathbb{R}^{d_h}
\tag{4.2}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

Since $\mathrm{elu}(x) > -1$ for all $x$, we have $\phi(x) > 0$.

### 4.3.2 Causal Formulation

For each position $i$:

$$
\text{cum}^{(i)} = \sum_{j \le i} \phi(K_j) V_j^\top \in \mathbb{R}^{d_h \times d_h}
\tag{4.3}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

$$
\text{den}^{(i)} = \sum_{j \le i} \phi(K_j) \in \mathbb{R}^{d_h}
\tag{4.4}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

$$
O^\text{lin}_i = \frac{\phi(Q_i) \cdot \text{cum}^{(i)}}{\phi(Q_i) \cdot \text{den}^{(i)} + \epsilon}
\tag{4.5}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

where $\epsilon = 10^{-5}$ prevents division by zero.

### 4.3.3 Non-Causal Formulation

For encoders (bidirectional), we use the same formulas without cumsum:

$$
O^\text{lin}_i = \frac{\phi(Q_i) \cdot \sum_j \phi(K_j) V_j^\top}{\phi(Q_i) \cdot \sum_j \phi(K_j) + \epsilon}
\tag{4.6}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

### 4.3.4 Padding Handling

We zero out padding keys before the cumsum:

$$
\phi(K_j) \leftarrow \phi(K_j) \cdot \text{valid}_j, \quad
V_j \leftarrow V_j \cdot \text{valid}_j
\tag{4.7}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

This ensures padding contributes nothing to numerator or denominator.

### 4.3.5 Numerical Stability

The cumsum is computed in FP32 to avoid overflow:

$$
\text{cum}^{(i)} = \text{cumsum}(\phi(K) V^\top).\text{float}()
\tag{4.8}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

We cast back to the model dtype after cumsum.

## 4.4 Fusion Gate

### 4.4.1 Parameterization

For each head $h$, a linear projection computes the gate logit:

$$
\text{logit}_{h,i} = w_h^\top x_i + b_h
\tag{4.9}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

where $x_i \in \mathbb{R}^d$ is the input hidden state at position $i$. The 
gate is:

$$
g_{h,i} = \sigma(\text{logit}_{h,i}) \in (0, 1)
\tag{4.10}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

### 4.4.2 Output

$$
O_{h,i} = g_{h,i} \cdot O^\text{dense}_{h,i} + (1 - g_{h,i}) \cdot O^\text{lin}_{h,i}
\tag{4.11}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

### 4.4.3 Initialization

We initialize $b_h = 1.8$, so $\sigma(1.8) \approx 0.858$ at initialization. 
This biases the model toward the **dense stream**, which matches the 
pretrained behavior.

**Table 4.1: Gate values at various biases.**

| $b_h$ | $\sigma(b_h)$ | Interpretation |
|---|---|---|
| $-2$ | 0.12 | Mostly linear |
| $0$ | 0.50 | Balanced |
| $1.8$ | 0.86 | **Mostly dense (default)** |
| $3$ | 0.95 | Almost purely dense |

### 4.4.4 Per-Head vs Global Gate

We use **per-head** gates. This allows different heads to prefer different 
streams, which matches empirical observations that some heads are local 
(better served by linear) and others are global (better served by dense).

## 4.5 Complete Forward Pass

**Algorithm 1:** HALO-ReGLA forward
```
Input:  hidden_states ∈ R^{B × L × d}
        attention_mask, position_embeddings
        is_causal
Output: output ∈ R^{B × L × d}

 1: Q, K, V ← proj(hidden_states)
 2: Q, K ← apply_rope(Q, K, position_embeddings)
 3: K, V ← repeat_kv(K, V, H / H_kv) if GQA
 4:
 5: # Dense stream
 6: S ← Q K^T / sqrt(d_h)
 7: S ← S + causal_mask if is_causal
 8: S ← S + pad_mask if attention_mask
 9: A_dense ← softmax(S)
10: O_dense ← A_dense @ V
11:
12: # Linear stream
13: Q_l ← elu(Q) + 1
14: K_l ← elu(K) + 1
15: K_l, V_mask ← apply_pad_mask(K_l, V, attention_mask)
16:
17: if is_causal:
18:     cum_kv ← cumsum(K_l ⊗ V_mask)         # (B, H, L, d_h, d_h)
19:     num ← (Q_l.unsqueeze(-2) @ cum_kv).squeeze(-2)
20:     cum_k ← cumsum(K_l)
21:     den ← (Q_l * cum_k).sum(-1, keepdim=True)
22: else:
23:     KV ← K_l^T @ V_mask
24:     Z ← K_l.sum(dim=2, keepdim=True)
25:     num ← Q_l @ KV
26:     den ← Q_l @ Z^T
27: O_lin ← num / den.clamp(min=1e-5)
28: O_lin ← O_lin * valid_mask
29:
30: # Gate and fuse
31: g ← sigmoid(gate(hidden_states))          # (B, H, L, 1)
32: O ← g * O_dense + (1 - g) * O_lin
33:
34: O ← O.reshape(B, L, H * d_h)
35: O ← out_proj(O)
36: return O
```

## 4.6 Complexity Analysis

| Component | Dense | Linear | HALO-ReGLA |
|---|---|---|---|
| Time | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L d_h^2)$ | $\mathcal{O}(L^2 d_h + L d_h^2)$ |
| Memory | $\mathcal{O}(L^2)$ | $\mathcal{O}(d_h^2)$ | $\mathcal{O}(L^2)$ |
| Causal cumsum | — | $\mathcal{O}(L d_h^2)$ | Same |

**Note.** HALO-ReGLA does **not** reduce total complexity because the dense 
stream is still computed. The benefit is in the **gate**: if the gate 
converges to 0 for many heads, we can drop those heads' dense computations 
post-training.

To realize speedup, one would:
1. Train with both streams.
2. Identify heads where $\bar{g}_h < \theta$ (e.g., 0.1).
3. Convert those heads to pure linear inference.

## 4.7 Parameter Analysis

| Component | Baseline | HALO-ReGLA |
|---|---|---|
| $W_Q$ | $d H d_h$ | Same |
| $W_K$ | $d H_{kv} d_h$ | Same |
| $W_V$ | $d H_{kv} d_h$ | Same |
| $W_O$ | $H d_h d$ | Same |
| **Gate** | 0 | $d H + H$ |
| LoRA | 0 | $2 r (d H d_h + \cdots)$ |

The gate adds $d H + H$ parameters (one linear per head). For $d = 768$, 
$H = 12$: **9,216 parameters**. Negligible.

## 4.8 Theoretical Notes

### 4.8.1 Approximation of Dense by Linear

**Claim.** If the attention distribution is smooth (no sharp peaks), the 
linear approximation is close to dense.

**Sketch.** Linear attention approximates softmax with kernel $\phi$. For 
bounded inputs and $d_h \to \infty$, PORF gives unbiased approximation. Our 
ELU kernel is heuristic but works well in practice.

### 4.8.2 Gate Convergence

**Claim.** The gate converges to a stationary point where:

$$
\frac{\partial \mathcal{L}}{\partial g_h} = 0 \Rightarrow O^\text{dense}_h = O^\text{lin}_h
\tag{4.12}

<!-- source: neuralops/attentions/halo_regla.py line 173 -->

<!-- source: neuralops/attentions/halo_regla.py line 173 -->
$$

or the gate is at a boundary ($g_h \to 0$ or $g_h \to 1$).

**Proof.** At optimum, $\partial \mathcal{L} / \partial g = (\nabla_O \mathcal{L}) \cdot (O^\text{dense} - O^\text{lin}) = 0$. 

### 4.8.3 Causality Guarantee

The causal prefix-sum formulation ensures position $i$ only attends to 
$j \le i$. This is because $\text{cum}^{(i)}$ depends only on $\{\phi(K_j), 
V_j : j \le i\}$. $\square$

---
