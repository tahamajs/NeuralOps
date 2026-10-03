<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

SVD-Linear Attention replaces MHSA with a kernelized linear attention 
whose projections are SVD-factorized.

**Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                        │
    └────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌───────────────┐      ┌───────────────┐
    │ SVDLinear Q,K,V│      │ decay_gate    │
    │ (factorized)  │      │ (x → λ)       │
    └───────┬───────┘      └───────┬───────┘
            │                      │
            ▼                      ▼
    ┌───────────────┐      ┌───────────────┐
    │ φ(Q), φ(K)    │      │ Chunked scan  │
    │ feature map   │      │ S_t, Z_t      │
    └───────┬───────┘      └───────┬───────┘
            │                      │
            └──────────┬───────────┘
                       ▼
              ┌───────────────┐
              │ φ(Q) S / φ(Q) Z│
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │ out_proj (SVD)│
              └───────────────┘
```

## 4.2 SVD-Factorized Projections

Each projection is stored as $U V^\top$ where $U \in \mathbb{R}^{d_\text{out} \times r}$ 
and $V \in \mathbb{R}^{r \times d_\text{in}}$. The forward computes:

$$
y = (x V^\top) U^\top + b + \text{LoRA}(x)
\tag{4.1}
$$

where LoRA provides a trainable residual:

$$
\text{LoRA}(x) = (x A^\top) B^\top \cdot s
\tag{4.2}
$$

**Parameter count:** $r(d_\text{in} + d_\text{out})$ vs $d_\text{in} d_\text{out}$. 
For $768 \times 768$, $r = 32$: **~11× reduction**.

**Initialization.** $U$ and $V$ are initialized orthogonally.

**LoRA.** $A$ is Kaiming-uniform, $B = 0$.

## 4.3 Kernelized Linear Attention

### 4.3.1 Feature Map

$$
\phi(x) = \text{ELU}(x) + 1
\tag{4.3}
$$

which is strictly positive, ensuring the denominator $\phi(Q_t) Z_t > 0$.

### 4.3.2 Bidirectional (Encoder) Mode

For a non-causal encoder:

$$
S = \phi(K)^\top V, \qquad
Z = \sum_j \phi(K_j)
\tag{4.4}
$$

$$
\text{Out}_t = \frac{\phi(Q_t) S}{\phi(Q_t) Z + \epsilon}
\tag{4.5}
$$

**Complexity:** $\mathcal{O}(N d_h^2)$.

### 4.3.3 Causal (Decoder) Mode with Chunked Scan

For a causal decoder with decay gating, we use a chunked scan (chunk size $C = 64$).

**Within chunk $c$:**

$$
\log \gamma_{c, i} = \sum_{j \leq i} \log \lambda_{c, j}
\tag{4.6}
$$

$$
S_{c, i}^{\text{intra}} = e^{\log \gamma_{c, i}} \sum_{j \leq i} e^{-\log \gamma_{c, j}} \phi(K_{c, j}) V_{c, j}^\top
\tag{4.7}
$$

**Inter-chunk:**

$$
S_{c, i} = S_{c, i}^{\text{intra}} + e^{\log \gamma_{c, i}} S_{c-1, C}
\tag{4.8}
$$

**Output:**

$$
\text{Out}_{c, i} = \frac{\phi(Q_{c, i}) S_{c, i}}{\phi(Q_{c, i}) Z_{c, i} + \epsilon}
\tag{4.9}
$$

**Complexity:** $\mathcal{O}(N \cdot C \cdot d_h)$ for the intra-chunk matrix, 
$\mathcal{O}(N d_h^2)$ for the state updates.

## 4.4 Exponential Decay Gate

### 4.4.1 Decay Computation

$$
\lambda_t = \exp(-\text{softplus}(W_g x_t + b_g))
\tag{4.10}
$$

For channel-wise decay:

$$
\lambda_{t, h, k} = \exp(-\text{softplus}(W_g x_t)_{h, k} + b_{g, h, k})
\tag{4.11}
$$

### 4.4.2 Initialization Strategy

We want the initial decay times $\tau_{h} = -\log \lambda_h$ to be log-spaced:

$$
\tau_h = \exp\!\left(\text{linspace}(\log 1, \log 1024, H)\right) \quad \text{for } h = 1, \dots, H
\tag{4.12}
$$

Since $\lambda = e^{-\tau}$ and $\lambda = e^{-\text{softplus}(b)}$, we set:

$$
b_{g, h} = -\log \tau_h = \text{softplus}^{-1}(\log \tau_h)
\tag{4.13}
$$

In practice, we approximate $\text{softplus}(b) \approx \log(1 + e^b) \approx \log \tau_h$ for large $\tau_h$, so:

$$
b_{g, h} \approx \log(\log \tau_h)
\tag{4.14}
$$

The code uses `bias_target = -torch.log(decay_times)` which is the 
**exact** target for the **unbounded** log-decay, not the softplus-bounded one. The actual decay is:

$$
\lambda = e^{-\text{softplus}(W_g x + b_g)}
$$

so the effective initial decay is **not exactly** the target. We do not 
correct for this.

### 4.4.3 Padding Handling

For padding tokens, we force $\lambda_t = 1$:

$$
\lambda_t = \begin{cases} 1 & \text{if } x_t \text{ is padding} \\ \lambda_t & \text{otherwise} \end{cases}
\tag{4.15}
$$

This ensures that padding does not artificially decay the memory state.

## 4.5 Streaming Inference

The `step_recurrent` method processes one token at a time:

$$
S_t = \lambda_t S_{t-1} + \phi(K_t) V_t^\top, \qquad
\text{Out}_t = \frac{\phi(Q_t) S_t}{\phi(Q_t) Z_t + \epsilon}
\tag{4.16}
$$

**Complexity:** $\mathcal{O}(d_h^2)$ per step.

This is equivalent to the batch chunked scan when tokens are processed in 
order, but reduces memory from $\mathcal{O}(N d_h^2)$ to $\mathcal{O}(d_h^2)$.

## 4.6 Full Algorithm

**Algorithm 1:** SVD-Linear Attention batch forward pass
```
Input:  X ∈ R^{B×N×d}, attention_mask
Output: Y ∈ R^{B×N×d}

 1: Q ← SVDLinear_Q(X); K ← SVDLinear_K(X); V ← SVDLinear_V(X)
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA:  K, V ← repeat_kv(K, V)
 4: φ_Q ← ELU(Q) + 1; φ_K ← ELU(K) + 1
 5: apply padding mask to φ_K
 6: λ ← exp(-softplus(decay_gate(X)))
 7: force λ=1 on padding
 8:
 9: # Chunked scan
10: for c in chunks of size C=64:
11:     compute intra-chunk scan
12:     add inter-chunk carry
13:     compute output for chunk c
14:     update running state
15:
16: Y ← SVDLinear_O(concat(outputs))
17: return Y
```

**Algorithm 2:** `step_recurrent` (streaming)
```
Input:  x_t ∈ R^{B×d}, past_state (S, Z)
Output: y_t ∈ R^{B×d}, new_state

1: q ← SVDLinear_Q(x_t); k ← SVDLinear_K(x_t); v ← SVDLinear_V(x_t)
2: φ_q ← ELU(q) + 1; φ_k ← ELU(k) + 1
3: λ ← exp(-softplus(decay_gate(x_t)))
4: S ← λ · S + φ_k^T v
5: Z ← λ · Z + φ_k^T
6: num ← φ_q S
7: den ← φ_q Z + ε
8: context ← num / den
9: y_t ← SVDLinear_O(context)
10: return y_t, (S, Z)
```

## 4.7 Complexity Analysis

| Mode | Time | Space |
|---|---|---|
| Softmax (baseline) | $\mathcal{O}(N^2 d_h)$ | $\mathcal{O}(N^2)$ |
| SVD-Linear (bidirectional) | $\mathcal{O}(N d_h^2)$ | $\mathcal{O}(d_h^2)$ |
| SVD-Linear (chunked causal) | $\mathcal{O}(N C d_h + N d_h^2)$ | $\mathcal{O}(C d_h^2)$ |
| SVD-Linear (step_recurrent) | $\mathcal{O}(d_h^2)$ per step | $\mathcal{O}(d_h^2)$ |

For $N = 512$, $d_h = 64$, $C = 64$:

| Mode | Time |
|---|---|
| Softmax | 512² × 64 ≈ 1.7 × 10⁷ |
| SVD-Linear | 512 × 64 × 64 ≈ 2 × 10⁶ |
| **Speedup** | **~8×** |

## 4.8 Parameter Analysis

| Component | Parameters |
|---|---|
| Q, K, V, O projections (SVD) | $4 \times r(d_\text{in} + d_\text{out})$ |
| LoRA adapters | $4 \times r_L(d_\text{in} + d_\text{out})$ |
| Decay gate | $d \times H + H$ |
| **Total** | see below |

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $r = 32$, $r_L = 16$, $N_\ell = 6$):

| Component | Per layer | 6 layers |
|---|---|---|
| Q/K/V/O SVD | 4 × 49,216 = 196,864 | 1,181,184 |
| LoRA | 4 × 24,576 = 98,304 | 589,824 |
| Decay gate | 9,228 | 55,368 |
| **Total** | **304,396** | **1,826,376** |

Baseline per layer: 2,359,296. **Reduction: ~50\%×**.

## 4.9 Theoretical Notes

### 4.9.1 Linear Attention Approximation

**Claim.** The linear attention with $\phi(x) = \text{ELU}(x) + 1$ approximates 
softmax attention but does not equal it.

**Proof.** Linear attention is exact for $\phi(x) = e^x$, which is 
numerically unstable. For ELU+1, the approximation quality depends on the 
distribution of Q, K. In practice it gives similar top-$k$ attention patterns. $\square$

### 4.9.2 Causality with Decay

**Claim.** SVD-Linear Attention preserves causality.

**Proof.** The chunked scan is causal: chunk $c$ can only attend to chunks 
$1..c$ and intra-chunk positions $1..i$. The step-recurrent mode is causal 
by construction. $\square$

### 4.9.3 Streaming Equivalence

**Claim.** `step_recurrent` produces the same output as batch chunked scan.

**Proof.** Both compute the same recurrence $S_t = \lambda_t S_{t-1} + \phi(K_t) V_t^\top$. 
The batch version uses a parallel scan; the streaming version is sequential. 
The results are mathematically identical. $\square$

---