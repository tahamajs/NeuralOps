<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

ASVD compresses each targeted linear projection in four phases:

1. **Extraction** (§4.2): extract $W_Q, W_K, W_V, W_O$ from the source 
   attention.
2. **Weighting** (§4.3): scale by activation energy.
3. **SVD + rank selection** (§4.4): decompose and select rank via energy.
4. **Validation + replacement** (§4.5): check Frobenius error, replace.

**Figure 4.1** shows the pipeline.

```
    ┌─────────────────────────────────────────────┐
    │  Phase 1: Extract W_Q, W_K, W_V, W_O        │
    │  (multi-arch: DistilBERT, RoBERTa, GPT-2,   │
    │   Qwen)                                     │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │  Phase 2: Weight by activation energy       │
    │  W_weighted = W · diag(√F)                  │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │  Phase 3: SVD + energy-based rank           │
    │  U, Σ, V = svd(W_weighted)                  │
    │  r = min{r : E(r) ≥ ε}                      │
    └──────────────────┬──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────┐
    │  Phase 4: Validate + replace                │
    │  err = ‖W - UΣV^T‖_F / ‖W‖_F                │
    │  if err ≤ τ: replace with ASVDLinear        │
    │  else: abort injection                      │
    └─────────────────────────────────────────────┘
```

## 4.2 Phase 1: Weight Extraction

ASVD extracts the four projections from the source attention via the 
`copy_weights` routine, which handles:

### 4.2.1 DistilBERT (Flat)

```python
extracted["q"] = src.q_lin.weight
extracted["k"] = src.k_lin.weight
extracted["v"] = src.v_lin.weight
extracted["o"] = src.out_lin.weight
```

### 4.2.2 RoBERTa (Nested)

```python
s = src.self
extracted["q"] = s.query.weight
extracted["k"] = s.key.weight
extracted["v"] = s.value.weight
# out_proj handled via set_identity (outer attention.output.dense applies it)
```

### 4.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
extracted["q"] = q_w
extracted["k"] = k_w
extracted["v"] = v_w
extracted["o"] = src.c_proj.weight.t()
```

### 4.2.4 Qwen (GQA + RoPE)

Direct extraction from `q_proj/k_proj/v_proj/o_proj`.

### 4.2.5 Pre-Flight Validation

Before replacing the attention module, ASVD constructs a new module with 
`copy_weights` and attempts extraction. If any essential projection is 
missing or has near-zero Frobenius norm, injection aborts — **no silent 
fallback to random initialization**.

## 4.3 Phase 2: Activation-Aware Weighting

### 4.3.1 The Weighting

Given weight $W$ and per-input-dimension activation energy $F \in \mathbb{R}^n$:

$$
W_\text{weighted} = W \cdot \text{diag}(\sqrt{F + \epsilon})
\tag{4.1}
$$

with $\epsilon = 10^{-6}$.

**Note.** Unlike AASVD (which uses the full covariance $C_X^{1/2}$), ASVD 
uses only the diagonal. This is simpler but ignores inter-feature 
correlations.

### 4.3.2 Energy Estimation

In practice, ASVD does not compute $F$ explicitly. Instead, it uses 
unweighted SVD as a first pass and then refines rank via energy 
thresholding. This is a design choice that differs from A3/AASVD.

**Alternative.** If calibration data is available, $F$ can be estimated 
from forward hooks and used to weight the SVD input. We provide this path 
as an option.

## 4.4 Phase 3: SVD and Rank Selection

### 4.4.1 SVD

Given $W$ (or $W_\text{weighted}$), compute:

$$
W = U \Sigma V^\top
\tag{4.2}
$$

The full SVD of a $768 \times 768$ matrix takes ~5 ms on RTX 4070.

### 4.4.2 Energy-Based Rank Selection

Compute cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m, n)} \sigma_i^2}
\tag{4.3}
$$

Select the minimal $r$ such that $E(r) \geq \varepsilon$:

$$
r_\text{energy} = \min\{r : E(r) \geq \varepsilon\}
\tag{4.4}
$$

### 4.4.3 Dynamic vs Fixed Rank

ASVD supports both:

- **Dynamic** (`dynamic_rank=True`): $r = \max(r_\text{min}, r_\text{energy})$
- **Fixed** (`dynamic_rank=False`): $r = \min(r_\text{target}, \min(m, n))$

Default is dynamic with $\varepsilon = 0.95$ and 
$r_\text{min} = 16$.

### 4.4.4 Error-Based Rank Expansion

If after selecting $r_\text{energy}$, the reconstruction error exceeds 
$\tau$, ASVD expands the rank until the error bound is met or the maximum 
rank is reached.

## 4.5 Phase 4: Validation and Replacement

### 4.5.1 Frobenius Error

$$
\text{err}(W, U, \Sigma, V) = \frac{\|W - U \Sigma V^\top\|_F}{\|W\|_F + 10^{-9}}
\tag{4.5}
$$

### 4.5.2 Failure Handling

If $\text{err} > \tau$ at maximum rank, ASVD **aborts injection**:

```python
raise RuntimeError(
    f"Relative Frobenius reconstruction error ({rel_err:.2%}) for projection "
    f"'{proj_name}' exceeds {self.error_threshold:.0%} threshold."
)
```

**Design choice.** This is different from A3 (which logs a warning and 
continues). ASVD's strict failure prevents silently broken models.

### 4.5.3 The Replacement Module

```python
class ASVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, lora_rank=0):
        super().__init__()
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.S = nn.Parameter(torch.empty(rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
    
    def forward(self, x):
        Vx = F.linear(x, self.V)          # (B, L, r)
        SVx = Vx * self.S                  # (B, L, r)
        out = F.linear(SVx, self.U)        # (B, L, d_out)
        if self.bias is not None:
            out = out + self.bias
        if self.lora_rank > 0:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + lora_out
        return out * self.output_scale
```

### 4.5.4 The `set_identity` Method

For nested encoders (RoBERTa/BERT), the output projection is applied 
externally. ASVD handles this by setting the out_proj to identity:

```python
def set_identity(self):
    with torch.no_grad():
        self.U.data.zero_()
        self.V.data.zero_()
        self.S.data.fill_(1.0)
        r = min(self.rank, self.in_features, self.out_features)
        self.U.data[:r, :r].fill_diagonal_(1.0)
        self.V.data[:r, :r].fill_diagonal_(1.0)
```

The module then acts as a pass-through, so the outer layer's 
`attention.output.dense` applies the actual projection.

## 4.6 Full Algorithm

**Algorithm 1:** ASVD compression
```
Input:  model M, rank r_target, ε, τ
        calibration data D (optional)
Output: compressed model M'

1:  for layer ℓ in attention_layers(M) do
2:      W_q, W_k, W_v, W_o ← extract(ℓ)
3:
4:      for W, target in [(W_q, q), (W_k, k), (W_v, v), (W_o, o)]:
5:          if target is nested out_proj:
6:              target.set_identity()
7:              continue
8:
9:          # Optional: compute activation energy F from D
10:         if use_activation_weighting and D is not None:
11:             W = W · diag(√F)
12:
13:         # SVD
14:         U, Σ, Vt ← svd(W)
15:         cum_energy ← cumsum(Σ²) / sum(Σ²)
16:         r ← min{r : cum_energy[r] ≥ ε}
17:         r ← max(r, r_min)
18:         r ← min(r, min(m, n))
19:
20:         # Validate
21:         W_rec ← U[:, :r] Σ[:r] Vt[:r, :]
22:         err ← ‖W - W_rec‖_F / ‖W‖_F
23:         if err > τ:
24:             raise RuntimeError(...)
25:
26:         target.set_svd_factors(U[:, :r], Vt[:r, :], Σ[:r])
27:     end for
28: end for
29: return M'
```

## 4.7 Complexity Analysis

| Phase | Time | Space |
|---|---|---|
| Extraction | $\mathcal{O}(1)$ | $\mathcal{O}(m n)$ |
| Weighting (optional) | $\mathcal{O}(N d^2)$ | $\mathcal{O}(d^2)$ |
| SVD | $\mathcal{O}(m n \min(m, n))$ | $\mathcal{O}(m n)$ |
| Validation | $\mathcal{O}(m n r)$ | $\mathcal{O}(m n)$ |
| **Total** | $\mathcal{O}(N_\ell d^3)$ | $\mathcal{O}(d^2)$ |

For DistilBERT: ~6 seconds.
For RoBERTa: ~20 seconds.

## 4.8 Parameter Analysis

Per layer:

| Component | Parameters |
|---|---|
| Original $W$ | $m n$ |
| ASVD | $r(m + n + 1)$ |
| ASVD + LoRA | $r(m + n + 1) + r_L(m + n)$ |

For $m = n = 768$:

| Rank | ASVD | ASVD + LoRA(16) | Ratio vs baseline |
|---|---|---|---|
| 16 | 24,592 | 49,168 | 12.0× |
| 32 | 49,152 | 73,728 | 8.0× |
| 64 | 98,304 | 122,880 | 4.8× |
| 128 | 196,608 | 221,184 | 2.7× |
| 256 | 393,216 | 417,792 | 1.4× |

**Aggregated (DistilBERT, 6 layers × 4 projections):**

| Rank | Total ASVD params | Compression ratio |
|---|---|---|
| 32 | 1,179,648 | **~57×** (attention block) |
| 64 | 2,359,296 | **~28×** |
| 128 | 4,718,592 | **~14×** |

## 4.9 Theoretical Notes

### 4.9.1 Optimality

**Claim.** The truncated SVD $W_r = U_{:, :r} \Sigma_{:r} V_{:r, :}^\top$ 
minimizes $\|W - W_r\|_F$ over all rank-$r$ matrices.

**Proof.** Eckart–Young theorem. $\square$

### 4.9.2 Energy Lower Bound

**Claim.** For energy threshold $\varepsilon$, the reconstruction error 
satisfies:

$$
\|W - W_r\|_F \leq (1 - \varepsilon) \|W\|_F
\tag{4.6}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2$. 
If $E(r) \geq \varepsilon$, then $\|W - W_r\|_F^2 \leq (1 - \varepsilon) \|W\|_F^2$. 
Taking square roots gives (4.6). $\square$

### 4.9.3 Causality

**Claim.** ASVD preserves causality for decoders.

**Proof.** Each projection is applied position-wise. The attention mask 
enforces $j \leq i$. Therefore, output at position $i$ depends only on 
inputs $1..i$. $\square$

### 4.9.4 Explicit Sigma Advantage

**Claim.** Keeping $\Sigma$ explicit enables rank reduction without 
re-factorization.

**Proof.** If the rank $r$ needs to be reduced to $r' < r$ after 
compression, we simply truncate:

$$
W \to W' = U_{:, :r'} \Sigma_{:r'} V_{:r', :}^\top
\tag{4.7}
$$

This is immediate; no additional SVD is required. $\square$

---