# 4. Method

## 4.1 Overview

HGRN2Layer replaces standard attention with a single element-wise recurrence. 
The pipeline is:

```
    Input X ∈ R^{B×L×d}
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
    f_gate  i_gate  v_proj
       │      │      │
       └──────┼──────┘
              │
        g = (1-f) * i * v
              │
              ▼
    ┌───────────────────┐
    │ Chunked scan      │
    │ (FP32, chunk=32)  │
    └─────────┬─────────┘
              │
              ▼
       h_ctx = scan(f, g)
              │
              ▼
    ┌───────────────────┐
    │ h_norm = LayerNorm│
    │ gated = (0.5v +   │
    │   0.5h_norm)*silu(z)│
    └─────────┬─────────┘
              │
              ▼
        out_proj
              │
              ▼
           Output Y
```

## 4.2 Gate Computation

**Forget gate:**

$$
\log f_t = -\text{softplus}(W_f x_t + b_f)
\tag{4.1}
$$

$$
\log f_t \leftarrow \text{clamp}(\log f_t, -3, -10^{-4})
\tag{4.2}
$$

$$
f_t = \exp(\log f_t) \in [e^{-3}, 1] \approx [0.05, 1]
\tag{4.3}
$$

**Initialization.** $W_f$ is normal $N(0, 0.01)$, $b_f = -2.5$ giving 
$\text{softplus}(-2.5) \approx 0.076$, so $f_0 \approx e^{-0.076} \approx 0.93$ 
— a slow decay.

**Input gate:**

$$
i_t = \sigma(W_i x_t + b_i)
\tag{4.4}
$$

**Value:**

$$
v_t = W_v x_t
\tag{4.5}
$$

**Input contribution:**

$$
g_t = (1 - f_t) \odot i_t \odot v_t
\tag{4.6}
$$

## 4.3 Chunked Parallel Scan

We partition the sequence into chunks of size $C = 32$. Within each chunk, 
we compute the intra-chunk recurrence in parallel:

**Step 1:** Compute cumulative log-forget within the chunk:

$$
A_k = \sum_{j=1}^{k} \log f_j
\tag{4.7}
$$

**Step 2:** Compute pairwise decay:

$$
D_{k,j} = \exp(A_k - A_j) \quad \text{for } j \leq k
\tag{4.8}
$$

Zero for $j > k$.

**Step 3:** Intra-chunk state:

$$
h_k^\text{intra} = \sum_{j=1}^{k} D_{k,j} \cdot g_j
\tag{4.9}
$$

**Step 4:** Combine with previous chunk state:

$$
h_k^\text{chunk} = h_\text{prev} \cdot \exp(A_k) + h_k^\text{intra}
\tag{4.10}
$$

**Step 5:** Clip for stability:

$$
h_k^\text{chunk} \leftarrow \text{clamp}(h_k^\text{chunk}, -30, 30)
\tag{4.11}
$$

**Step 6:** Update $h_\text{prev} = h_C^\text{chunk}$ and proceed to next chunk.

## 4.4 Bidirectional Recurrence

For encoders, we compute both forward and backward scans:

$$
h_t^\text{fwd} = \text{scan}(f_{1:L}, g_{1:L})_t
\tag{4.12}
$$

$$
h_t^\text{bwd} = \text{scan}(\text{flip}(f), \text{flip}(g))_{L-t+1}
\tag{4.13}
$$

$$
h_t = h_t^\text{fwd} + h_t^\text{bwd}
\tag{4.14}
$$

For decoders (causal), we use only the forward scan.

## 4.5 Output Modulation

The raw state $h_t$ is normalized and gated:

$$
h_t^\text{norm} = \text{LayerNorm}(h_t)
\tag{4.15}
$$

$$
\text{combined}_t = 0.5 \cdot v_t + 0.5 \cdot h_t^\text{norm}
\tag{4.16}
$$

$$
\text{gated}_t = \text{combined}_t \odot \text{SiLU}(z_t)
\tag{4.17}
$$

where $z_t = W_g x_t + b_g$ is a gating projection initialized with $b_g = 1.28$.

**Why combine with $v_t$?** Directly using $h_t$ can cause the output to 
drift far from the current value $v_t$. Averaging $0.5 v_t + 0.5 h_t$ 
preserves a self-value path and stabilizes training.

**Why SiLU gating?** A nonlinear gating on the combined signal allows the 
model to suppress or amplify the recurrence.

## 4.6 Output Projection

$$
\text{out} = W_o \text{gated} + b_o + \text{LoRA}(\text{gated})
\tag{4.18}
$$

## 4.7 Algorithm

**Algorithm 1:** HGRN2Layer forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
        chunk_size C = 32
Output: Y ∈ R^{B×L×d}

 1: f ← exp(clamp(-softplus(W_f X + b_f), -3, -1e-4))
 2: i ← sigmoid(W_i X + b_i)
 3: v ← W_v X
 4: z ← W_g X + b_g
 5: g ← (1 - f) ⊙ i ⊙ v
 6: h_fwd ← chunked_scan(f, g, C)
 7: if not is_decoder:
 8:     h_bwd ← flip(chunked_scan(flip(f), flip(g), C))
 9:     h ← h_fwd + h_bwd
10: else:
11:     h ← h_fwd
12: h_norm ← LayerNorm(h)
13: gated ← (0.5 v + 0.5 h_norm) ⊙ SiLU(z)
14: Y ← W_o gated + b_o + LoRA(gated)
15: return Y
```

## 4.8 Complexity Analysis

For a single layer:

| Component | Time | Memory |
|---|---|---|
| Forget gate | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Input gate | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Value | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Gate $z$ | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| Chunked scan | $\mathcal{O}(L d)$ | $\mathcal{O}(C^2 + L d)$ |
| Output | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |

Compare to attention: $\mathcal{O}(L^2 d)$. Our method is **linear** in $L$.

## 4.9 Parameter Analysis

**HGRN2Layer does NOT reduce parameter count.**

| Component | Baseline | HGRN2Layer |
|---|---|---|
| Attention Q/K/V/O | $4 d^2$ | 0 |
| Forget gate | 0 | $d^2 + d$ |
| Input gate | 0 | $d^2 + d$ |
| Value | 0 | $d^2$ |
| Gate $z$ | 0 | $d^2 + d$ |
| StateNorm | 0 | $2d$ |
| Output | 0 | $d^2 + d$ |
| LoRA | 0 | $4 r_L d$ |
| **Total per layer** | $4 d^2$ | $5 d^2 + \mathcal{O}(d)$ |

For $d = 768$, $r_L = 16$:

| Config | Params |
|---|---|
| Baseline | 2,359,296 |
| HGRN2Layer | ~2,949,120 |
| **Increase** | **+25%** |

## 4.10 Training

Trainable parameters:
- Forget gate $W_f, b_f$
- Input gate $W_i, b_i$
- Gate $W_g, b_g$
- LoRA on $W_v$ and $W_o$

Frozen:
- $W_v, W_o$ base weights
- All LoRA base weights from pretrained attention

**Trainable count:** for $d = 768$, $r_L = 16$:
- Gates: $\sim 2.4M$
- LoRA: $\sim 98K$
- Total: $\sim 2.5M$ per layer

---