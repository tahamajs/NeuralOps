<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

RPC replaces the standard attention score with a sketched version:

1. **Project** $Q, K$ to a lower-dimensional subspace using a fixed random 
   matrix $\Omega$.
2. **Compute** scores in the projected space.
3. **Apply** softmax and multiply by $V$ (unprojected).

**Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────────┐
    │  Input X ∈ R^{B×L×d}                        │
    └────────────────┬───────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
    ┌──────────┐             ┌──────────┐
    │ Q, K, V  │             │ Ω ∈ R^   │
    │ proj     │             │ {d_h×r}  │
    └────┬─────┘             └────┬─────┘
         │                        │
         ▼                        ▼
    ┌──────────────────────────────────────┐
    │  Sketch: Q̃ = QΩ, K̃ = KΩ              │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Score: Q̃ K̃^T / √r                   │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Softmax + @ V                        │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Output projection                    │
    └──────────────────────────────────────┘
```

## 4.2 Random Projection Matrix

### 4.2.1 Construction

For each head $h$, RPC samples:

$$
\Omega_h \in \mathbb{R}^{d_h \times r}, \qquad
(\Omega_h)_{ij} \sim \mathcal{N}\!\left(0, \frac{1}{d_h}\right)
\tag{4.1}
$$

This is stored as a **buffer** (not a parameter):

```python
self.register_buffer(
    "proj", torch.randn(self.num_heads, self.head_dim, self.rank) * (self.head_dim**-0.5)
)
```

### 4.2.2 Why Fixed?

The random projection is **fixed** (not trained) for three reasons:

1. **JL guarantee:** Only random matrices have the JL property. Training 
   could destroy it.
2. **No parameter cost:** Buffers are not parameters, so they don't increase 
   model size in the usual sense.
3. **Reproducibility:** Fixed projection makes results reproducible across 
   training runs.

### 4.2.3 Head-Wise vs Shared

RPC uses **head-wise** projections $\Omega_h$, so each head learns (implicitly) 
its own subspace. This is different from using a single $\Omega$ shared 
across heads.

## 4.3 Forward Pass

### 4.3.1 Standard Attention

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_h}}, \qquad
P = \text{softmax}(S), \qquad
\text{out} = P V
\tag{4.2}
$$

### 4.3.2 RPC Attention

$$
\tilde{Q}_i = Q_i \Omega_h, \qquad
\tilde{K}_j = K_j \Omega_h
\tag{4.3}
$$

$$
S_{ij}^{RPC} = \frac{\tilde{Q}_i \tilde{K}_j^\top}{\sqrt{r}}, \qquad
P^{RPC} = \text{softmax}(S^{RPC}), \qquad
\text{out} = P^{RPC} V
\tag{4.4}
$$

**Note.** $V$ is not sketched; only $Q$ and $K$.

### 4.3.3 Causality

For decoders, the causal mask is applied in the sketched space:

$$
S^{RPC} = S^{RPC} + M_{causal}
\tag{4.5}
$$

where $M_{ij} = -\infty$ for $j > i$.

## 4.4 Complexity Analysis

| Component | Standard | RPC |
|---|---|---|
| Q/K/V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d^2)$ |
| Sketch Q, K | — | $\mathcal{O}(L d_h r H)$ |
| Score | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 r H)$ |
| Softmax | $\mathcal{O}(L^2 H)$ | $\mathcal{O}(L^2 H)$ |
| @ V | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 d_h H)$ |
| **Total** | $\mathcal{O}(L^2 d_h H)$ | $\mathcal{O}(L^2 r H + L d_h r H)$ |

**Speedup** for $L \gg d_h$:

$$
\frac{\text{Standard}}{\text{RPC}} \approx \frac{d_h}{r}
\tag{4.6}
$$

For $d_h = 64, r = 32$: theoretical **2× speedup**.

For $d_h = 64, r = 16$: theoretical **4× speedup**.

**Memory.** RPC **does not reduce** memory; $S^{RPC}$ is still $L \times L$.

## 4.5 Parameter Analysis

RPC adds:

| Component | Parameters |
|---|---|
| LoRA Q | $r_L(d + H d_h)$ |
| LoRA K | $r_L(d + H_{kv} d_h)$ |
| LoRA V | $r_L(d + H_{kv} d_h)$ |
| LoRA O | $r_L(H d_h + d)$ |
| Random proj | **0** (buffer, not parameter) |

For DistilBERT ($d = 768$, $H = 12$, $d_h = 64$, $r_L = 16$, $N_\ell = 6$):

| Component | Per layer | 6 layers |
|---|---|---|
| LoRA (4 projections) | 98,304 | 589,824 |
| Proj buffer | 0 | 0 |
| **Total** | **98,304** | **589,824** |

**+48.5\%%** parameter increase on DistilBERT.

## 4.6 Full Algorithm

**Algorithm 1:** RPC forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask, Ω (fixed)
Output: Y ∈ R^{B×L×d}

1:  Q ← W_Q X + LoRA_Q(X)          ▷ (B, H, L, d_h)
2:  K ← W_K X + LoRA_K(X)
3:  V ← W_V X + LoRA_V(X)
4:  if RoPE: (Q, K) ← Rotary(Q, K)
5:  if GQA:  K, V ← repeat_kv(K, V)
6:
7:  # Sketch
8:  Q̃ ← einsum("bhld,hdr->bhlr", Q, Ω)
9:  K̃ ← einsum("bhld,hdr->bhlr", K, Ω)
10:
11: # Score
12: S ← Q̃ K̃^T / √r                  ▷ (B, H, L, L)
13: if causal: S ← S + causal_mask
14: if mask:   S ← S + pad_mask
15:
16: # Attend
17: P ← softmax(S)
18: out ← P V                        ▷ (B, H, L, d_h)
19: out ← concat_heads(out)
20: Y ← W_O out + LoRA_O(out)
21: return Y
```

## 4.7 Theoretical Notes

### 4.7.1 JL Error Bound

**Claim.** For $r \geq 8 \log(L) / \epsilon^2$, with probability $1 - L^{-2}$:

$$
|S^{RPC}_{ij} - S_{ij}| \leq \epsilon \|Q_i\|_2 \|K_j\|_2 / \sqrt{r}
\tag{4.7}
$$

**Proof sketch.** Direct consequence of JL applied to the concatenated 
vector $[Q_i; K_j]$. Full derivation in Appendix A.1. $\square$

### 4.7.2 Softmax Stability

**Claim.** The sketched scores $S^{RPC}$ have bounded variance:

$$
\text{Var}[S^{RPC}_{ij}] \approx \frac{1}{r} \|Q_i\|_2^2 \|K_j\|_2^2
\tag{4.8}
$$

**Proof.** Each entry $\tilde{Q}_{i,k}$ has variance $\|Q_i\|_2^2 / d_h \cdot d_h = \|Q_i\|_2^2$... 
full derivation in Appendix A.2. $\square$

### 4.7.3 Causality

**Claim.** RPC preserves causality for decoders.

**Proof.** The causal mask is applied after sketching. Since sketching is 
position-independent, it does not affect the causal structure. $\square$

---