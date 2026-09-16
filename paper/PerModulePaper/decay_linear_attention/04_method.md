# 4. Method

## 4.1 Overview

DecayLinearAttention replaces standard attention with a **chunked, 
data-dependent linear recurrence**. For each head, we learn a per-token decay 
$\lambda_{h,t}$ and propagate a running state:

$$
S_t = \lambda_{h,t} S_{t-1} + \phi(k_t) v_t^\top
\tag{4.1}
$$

$$
z_t = \lambda_{h,t} z_{t-1} + \phi(k_t)
\tag{4.2}
$$

The output:

$$
O_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t + \epsilon}
\tag{4.3}
$$

**Figure 4.1** shows the pipeline.

```
    X ∈ R^{B×L×d}
         │
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
    W_q W_k W_v  decay_gate
    │   │   │    │
    ▼   ▼   ▼    ▼
    Q   K   V   λ_t
    │   │   │    │
    ▼   ▼   ▼    │
    φ(Q) φ(K) V  │
    │   │   │    │
    └───┼───┼────┘
        ▼   ▼
    Chunked Scan (S, z)
        │
        ▼
    O = φ(Q) S / (φ(Q) z)
        │
        ▼
    W_o
        │
        ▼
    Output Y
```

## 4.2 Data-Dependent Decay

For each head $h$ and token $t$:

$$
\tilde{\lambda}_{h,t} = W_g x_t + b_g
\tag{4.4}
$$

$$
\lambda_{h,t} = \sigma\!\left(\tilde{\lambda}_{h,t} + \ell_h\right)
\tag{4.5}
$$

where $\ell_h$ is a **learned scalar** per head.

**Initialization.**
- $W_g = 0$ (no input dependence initially).
- $b_g = 0$.
- $\ell_h$ is initialized to linearly spaced values in 
  $[{{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}]$:

  ```python
  self.log_decay = nn.Parameter(torch.linspace(-0.5, 0.0, num_heads))
  ```

**Result.** At step 0, different heads start with different horizons:
- Head 0: $\lambda \approx \sigma({{LOG_DECAY_MIN}}) \approx {{LAMBDA_MIN}}$
- Head $H-1$: $\lambda \approx \sigma({{LOG_DECAY_MAX}}) \approx {{LAMBDA_MAX}}$

This head diversity is analogous to multi-scale temporal receptive fields in 
convolutional networks.

## 4.3 ELU+1 Feature Map

$$
\phi(x) = \text{elu}(x) + 1
\tag{4.6}
$$

Applied element-wise to Q, K.

**Why ELU+1?**
- **Positive:** Ensures $z_t > 0$.
- **Cheap:** No exponential or trigonometric functions.
- **Established:** Used by Linear Transformer (Katharopoulos et al., 2020).

## 4.4 Chunked Scan

For a chunk of size $C$ starting at position $s$:

### 4.4.1 Cumulative Log-Decay

Within the chunk, we compute:

$$
\Lambda_i = \sum_{p=s}^{s+i} \log \lambda_{h,p}, \quad i \in [0, C)
\tag{4.7}
$$

via `torch.cumsum`.

### 4.4.2 Intra-Chunk Attention

For positions $i, j$ in the chunk with $j \leq i$:

$$
D_{i, j}^\text{intra} = \exp(\Lambda_i - \Lambda_j)
\tag{4.8}
$$

$$
A_{i, j}^\text{intra} = \phi(q_i)^\top \phi(k_j) \cdot D_{i, j}^\text{intra}
\tag{4.9}
$$

with $A_{i, j}^\text{intra} = 0$ for $j > i$ (causality).

Output (numerator):

$$
\text{num}_i^\text{intra} = \sum_{j \leq i} A_{i, j}^\text{intra} \cdot v_j
\tag{4.10}
$$

Normalizer (denominator):

$$
\text{denom}_i^\text{intra} = \sum_{j \leq i} A_{i, j}^\text{intra}
\tag{4.11}
$$

### 4.4.3 Inter-Chunk Contribution

Let $S_\text{prev}$, $z_\text{prev}$ be the state before the chunk. Define:

$$
D_i^\text{inter} = \exp(\Lambda_i)
\tag{4.12}
$$

Then:

$$
\text{num}_i^\text{inter} = \phi(q_i)^\top S_\text{prev} \cdot D_i^\text{inter}
\tag{4.13}
$$

$$
\text{denom}_i^\text{inter} = \phi(q_i)^\top z_\text{prev} \cdot D_i^\text{inter}
\tag{4.14}
$$

### 4.4.4 Output

$$
O_i = \frac{\text{num}_i^\text{intra} + \text{num}_i^\text{inter}}{\text{denom}_i^\text{intra} + \text{denom}_i^\text{inter} + \epsilon}
\tag{4.15}
$$

### 4.4.5 State Update

After the chunk, the state is updated for the next chunk:

$$
S_\text{new} = \exp(\Lambda_C) S_\text{prev} + \sum_{i=0}^{C-1} \exp(\Lambda_C - \Lambda_i) \phi(k_i) v_i^\top
\tag{4.16}
$$

$$
z_\text{new} = \exp(\Lambda_C) z_\text{prev} + \sum_{i=0}^{C-1} \exp(\Lambda_C - \Lambda_i) \phi(k_i)
\tag{4.17}
$$

where $\Lambda_C$ is the cumulative log-decay at the end of the chunk.

## 4.5 Bidirectional Mode

For encoders, we run the chunked scan twice:

$$
O^\text{fwd} = \text{Scan}(Q, K, V, \lambda) \quad \text{(left to right)}
$$

$$
O^\text{bwd} = \text{Scan}(Q, K, V, \lambda) \quad \text{(right to left)}
$$

where the backward scan flips the sequence, runs the forward scan, then 
flips back.

Final output:

$$
O = \frac{1}{2} (O^\text{fwd} + O^\text{bwd})
\tag{4.18}
$$

## 4.6 Algorithm

**Algorithm 1:** Chunked decay scan
```
Input:  Q, K, V ∈ R^{B×H×L×d}, λ ∈ R^{B×H×L}
        chunk_size C = 32
Output: O ∈ R^{B×H×L×d}

 1: log_λ ← log(clamp(λ, 1e-4, 1-1e-4))
 2: Λ ← cumsum(log_λ, dim=-1)
 3: Pad L to multiple of C
 4: Reshape into (B, H, num_chunks, C, d)
 5: S ← zeros(B, H, d, d)      ▷ Inter-chunk state
 6: z ← zeros(B, H, d, 1)
 7: for c in 0..num_chunks-1:
 8:     Λ_c ← Λ[:,:,c*C:(c+1)*C]         ▷ Cumulative in chunk
 9:     D_intra ← exp(Λ_c[:,:,:,None] - Λ_c[:,:,None,:])
10:     D_intra = tril(D_intra)             ▷ Causal
11:     A_intra ← Q_c @ K_c^T * D_intra
12:     num_intra ← A_intra @ V_c
13:     denom_intra ← A_intra.sum(-1)
14:
15:     D_inter ← exp(Λ_c)  ▷ (B, H, C, 1)
16:     num_inter ← Q_c @ S * D_inter
17:     denom_inter ← Q_c @ z * D_inter
18:
19:     O_c ← (num_intra + num_inter) / (denom_intra + denom_inter + ε)
20:
21:     # Update state for next chunk
22:     Λ_end ← Λ_c[:, :, -1:]  ▷ (B, H, 1)
23:     decay_to_end ← exp(Λ_end - Λ_c)  ▷ (B, H, C, 1)
24:     k_to_end ← K_c * decay_to_end
25:     S ← S * exp(Λ_end).squeeze(-1) + k_to_end^T @ V_c
26:     z ← z * exp(Λ_end).transpose + k_to_end.sum(dim=-2)
27: return O
```

## 4.7 Complexity Analysis

For a single layer with $L$ tokens:

| Component | Time | Memory |
|---|---|---|
| Projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Decay gate | $\mathcal{O}(L d H)$ | $\mathcal{O}(L H)$ |
| Chunk cumsum | $\mathcal{O}(L)$ | $\mathcal{O}(L)$ |
| Intra-chunk | $\mathcal{O}(L \cdot C)$ | $\mathcal{O}(L \cdot C)$ |
| Inter-chunk | $\mathcal{O}(L \cdot d_h)$ | $\mathcal{O}(d_h^2)$ |
| State update | $\mathcal{O}(L \cdot d_h^2 / C)$ | $\mathcal{O}(d_h^2)$ |
| **Total** | $\mathcal{O}(L d^2 + L \cdot C + L \cdot d_h^2 / C)$ | $\mathcal{O}(L d + d_h^2)$ |

**Key observation:** The state memory is $\mathcal{O}(d_h^2)$, independent 
of $L$. In contrast, CHELA-SLAY's causal cumsum uses $\mathcal{O}(L \cdot 
2F \cdot d_h)$.

For $d_h = 64$, $C = 32$:

| Metric | DecayLinear | CHELA-SLAY causal |
|---|---|---|
| State memory | $64^2 = 4096$ floats = **16 KB** | $128 \times 256 \times 64$ per layer = **8 GB** |

**500,000× less memory** at $L = 128$.

## 4.8 Parameter Analysis

| Component | Baseline | DecayLinear |
|---|---|---|
| Q, K, V, O | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA | 0 | $8 r_L d$ |
| Decay gate $W_g$ | 0 | $d H$ |
| Decay gate $b_g$ | 0 | $H$ |
| Per-head $\ell$ | 0 | $H$ |

For $d = 768$, $H = 12$, $r_L = 16$:

| Component | Params |
|---|---|
| LoRA | 98,304 |
| $W_g$ | 9,216 |
| $b_g$ | 12 |
| $\ell$ | 12 |
| **Total added** | ~108,000 |

**Relative to baseline attention:** +4.6%.

## 4.9 Step-0 Preservation

**Claim.** At initialization ($W_g = 0$, $b_g = 0$, LoRA $B = 0$), the 
effective decay is **constant per head** at value $\sigma(\ell_h)$, and 
the linear attention output is a specific linear combination of V.

**This is NOT equivalent to softmax attention** — the linear stream computes 
a different output. But the LoRA adapters can correct the discrepancy 
during recovery.

## 4.10 Weight Transfer

Base weights $W_Q, W_K, W_V, W_O$ are copied from the pretrained attention. 
The decay gate and per-head bias are initialized as above; the LoRA adapters 
are zero.
```

---