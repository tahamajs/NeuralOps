# 2. Related Work

## 2.1 State-Space Models

### 2.1.1 Classical SSMs

**S4** (Gu et al., 2022) introduced structured state-space models with 
diagonal plus low-rank A matrices, achieving $\mathcal{O}(L)$ complexity. 
**S5** (Smith et al., 2023) parallelized the SSM scan across time.

### 2.1.2 Selective SSMs (Mamba)

**Mamba** (Gu & Dao, 2023) introduced **input-dependent** state transitions:

$$
h_t = \bar{A}(x_t) h_{t-1} + \bar{B}(x_t) x_t
\tag{2.1}
$$

where $\bar{A}, \bar{B}$ depend on the input $x_t$. This selective mechanism 
makes the SSM behave like a learned, content-aware recurrence.

**Our distinction.** Our implementation **does not** perform a true selective 
scan. We use learned exponential decay as a **proxy** for the selectivity, 
implemented as a pairwise softmax over positions.

## 2.2 Efficient Attention

**Linformer** (Wang et al., 2020) projects K, V to a fixed rank. 
**Performer** (Choromanski et al., 2021) uses random Fourier features. 
**Longformer** (Beltagy et al., 2020) uses sparse local windows.

**Our distinction.** We exploit SSM-inspired decay rather than a fixed 
sequence projection or kernel approximation.

## 2.3 Attention-Replacement Modules

**Hyena** (Poli et al., 2023) uses long convolutions. **RWKV** (Peng et al., 
2023) uses a linear recurrence with time-decay. **RetNet** (Sun et al., 2023) 
uses a retention mechanism with decay.

**Our distinction.** We operate on pretrained attention weights, reusing 
$W_Q, W_K, W_V, W_O$ in the SSM structure.

## 2.4 Position of Our Work

| Method | Complexity | Reuses pretrained? | Causal | Bidirectional |
|---|---|---|---|---|
| Mamba | $\mathcal{O}(L)$ | No | Yes | Yes (Bi-Mamba) |
| Hyena | $\mathcal{O}(L \log L)$ | No | Yes | Yes |
| RWKV | $\mathcal{O}(L)$ | No | Yes | No |
| **Ours** | **$\mathcal{O}(L^2 D)$** | **Yes** | **Yes** | **Yes** |

**Key distinction.** We reuse pretrained attention weights in an SSM-inspired 
formulation, trading complexity for compatibility with pretrained transformers.

---

## Comparison Table

| Method | Decay | Learned? | Weight reuse | Causal |
|---|---|---|---|---|
| Mamba | Selective | ✓ | No | ✓ |
| RWKV | Fixed | Partial | No | ✓ |
| RetNet | Fixed | ✓ | No | ✓ |
| **Ours** | Learned per-dim | ✓ | **✓** | ✓ |

---