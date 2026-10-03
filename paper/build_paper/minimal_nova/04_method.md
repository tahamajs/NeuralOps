<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

MinimalNOVA replaces the four attention projections of each transformer layer 
with **SVD-factorized linear modules**:

$$
W \to (U, V), \qquad
y = (x V^\top) U^\top + \text{LoRA}(x) + b
\tag{4.1}
$$

The factorization is done independently per projection. Rank is chosen 
adaptively per layer based on spectral energy.

**Figure 4.1** shows the pipeline.

```
       ┌──────────────────────────────────────────┐
       │  Input weight W ∈ R^{d_out × d_in}       │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Compute SVD: W = U Σ V^T                │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Compute cumulative energy E(r)          │
       │  r_eff = min(r_max, argmin E(r) ≥ η)     │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Truncate to r_eff, symmetrize Σ         │
       │  U_r = U[:, :r] · √Σ_r                   │
       │  V_r = √Σ_r · V[:r, :]^T                 │
       └────────────────┬─────────────────────────┘
                        │
                        ▼
       ┌──────────────────────────────────────────┐
       │  Replace Linear with FactorisedLinear    │
       │  Freeze U, V; add trainable LoRA         │
       └──────────────────────────────────────────┘
```

## 4.2 The `FactorisedLinear` Module

### 4.2.1 Forward Pass

The module computes:

$$
y = \text{up}(\text{down}(x)) + \text{LoRA}(x) + b
\tag{4.2}
$$

where:

- `down: nn.Linear(d_in, r)` — down-projection, weights $V_r \in \mathbb{R}^{r \times d_\text{in}}$
- `up: nn.Linear(r, d_out)` — up-projection, weights $U_r \in \mathbb{R}^{d_\text{out} \times r}$
- `LoRA(x) = (x A^\top) B^\top \cdot s` — optional adapter

### 4.2.2 Parameter Count

| Configuration | Parameters |
|---|---|
| Original `nn.Linear` | $d_\text{in} d_\text{out}$ |
| `FactorisedLinear` at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| `FactorisedLinear` + LoRA($r_L$) | $r(d_\text{in} + d_\text{out}) + r_L(d_\text{in} + d_\text{out})$ |

**Example** for a $768 \times 768$ attention projection:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| r=64 | 98,304 | **6.00×** |
| r=128 | 196,608 | 3.00× |
| r=64 + LoRA r=16 | 122,880 | 4.80× |

## 4.3 Energy-Adaptive Rank Selection

### 4.3.1 Algorithm

Given weight $W$ and hyperparameters $r_\text{max}$ (cap) and $\eta$ (energy):

1. Compute SVD: $W = U \Sigma V^\top$.
2. Compute cumulative energy: $E(r) = \sum_{i \leq r} \sigma_i^2 / \sum_i \sigma_i^2$.
3. Find the smallest $r$ with $E(r) \geq \eta$:
   $$
   r_\text{energy} = \left|\left\{i : E(i) < \eta\right\}\right| + 1
   \tag{4.3}
   $$
4. Cap the rank:
   $$
   r_\text{eff} = \min(r_\text{max}, r_\text{energy}, \min(d_\text{in}, d_\text{out}))
   \tag{4.4}
   $$

### 4.3.2 Why Energy?

The energy criterion captures **how concentrated the spectrum is**. 
Consider two extreme cases:

- **Rank-1 matrix** ($\sigma_1 > 0$, $\sigma_{i>1} = 0$): $r_\text{energy} = 1$ 
  for any $\eta > 0$. Very compressible.
- **Identity matrix** ($\sigma_i = 1$ for all $i$): $E(r) = r/d$, so 
  $r_\text{energy} = \lceil \eta d \rceil$. Less compressible.

Real transformer weights lie between these extremes, with power-law or 
exponential decay. The energy criterion adapts to each matrix's specific 
spectrum.

### 4.3.3 Comparison to Fixed Rank

| Criterion | Pros | Cons |
|---|---|---|
| Fixed rank | Simple | Ignores heterogeneity |
| Energy | Adaptive | No calibration |
| Fisher-weighted | Activation-aware | Needs data |

MinimalNOVA uses energy, which is **training-free** and requires no 
calibration data.

## 4.4 Weight Factorization

### 4.4.1 Symmetric Distribution

We distribute $\Sigma^{1/2}$ between the two factors:

$$
U_r = U_{:, :r} \cdot \Sigma_{:r, :r}^{1/2}, \qquad
V_r = \Sigma_{:r, :r}^{1/2} \cdot V_{:r, :}^\top
\tag{4.5}
$$

This gives $W_r = U_r V_r$ with balanced factor magnitudes, which improves 
training stability.

### 4.4.2 Storage in Linear Modules

The framework stores $V_r$ in `down.weight` and $U_r$ in `up.weight`:

```python
self.down.weight.data[:r, :] = V_r  # (r, d_in)
self.up.weight.data[:, :r] = U_r  # (d_out, r)
```

For $r < \text{target rank}$, the remaining rows/columns are zero.

## 4.5 Weight Transfer from Pretrained Attention

MinimalNOVA extracts $W_Q, W_K, W_V, W_O$ from the pretrained attention 
module. This is nontrivial because different architectures store them 
differently.

### 4.5.1 Extraction Path (per architecture)

| Architecture | $W_Q$ | $W_K$ | $W_V$ | $W_O$ |
|---|---|---|---|---|
| DistilBERT | `q_lin` | `k_lin` | `v_lin` | `out_lin` |
| RoBERTa | `self.query` | `self.key` | `self.value` | `output.dense` |
| GPT-2 | `c_attn[:d]` | `c_attn[d:2d]` | `c_attn[2d:]` | `c_proj` |
| Qwen3.5 | `q_proj` | `k_proj` | `v_proj` | `o_proj` |

### 4.5.2 GPT-2 Fused c_attn

GPT-2 fuses $Q, K, V$ into one `c_attn`:

$$
W_{c\_attn} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix} \in \mathbb{R}^{3d \times d}
\tag{4.6}
$$

(Conv1D stores this transposed.) MinimalNOVA splits:

```python
w = src.c_attn.weight.t()  # (3d, d)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 4.5.3 RoBERTa Nested

The output projection $W_O$ is at `attention.output.dense`, **outside** the 
inner attention:

```python
# Extract Q, K, V from inner self-attention
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight

# Extract W_O from outer layer
w_o = src.output.dense.weight
```

When replacing the inner attention, the outer `output.dense` remains in 
the model and applies the output projection **after** our module returns. 
So we set `is_nested_encoder = True` and skip `out_proj` in the forward 
pass.

### 4.5.4 GQA (Qwen)

For GQA, $H_{kv} < H$. The K and V projections have smaller output 
dimensions:

$$
W_K \in \mathbb{R}^{H_{kv} d_h \times d}, \quad
W_V \in \mathbb{R}^{H_{kv} d_h \times d}
\tag{4.7}
$$

MinimalNOVA factorizes each at its natural dimension. During forward, 
`repeat_kv` expands K and V to match H query heads.

## 4.6 Validation and Fallback

### 4.6.1 Frobenius Validation

After factorization, MinimalNOVA does not explicitly validate (unlike A3). 
Instead, it relies on **weight copy verification**:

```python
for proj in ["q", "k", "v"]:
    w = extracted[proj][0]
    if w is not None:
        norm = torch.linalg.norm(w.float())
        if norm < 1e-4:
            raise RuntimeError(f"Near-zero Frobenius norm on {proj}")
```

This catches catastrophic extraction failures but does not bound the 
**factorization error**. This is a limitation (§10.7).

### 4.6.2 Fallback

If `copy_weights` raises, `inject_minimal_nova` catches and returns `None`. 
The original attention is preserved.

## 4.7 LoRA Recovery

After factorization, base weights are frozen:

```python
def freeze_base_weights(self):
    self.down.weight.requires_grad = False
    self.up.weight.requires_grad = False
    if self.up.bias is not None:
        self.up.bias.requires_grad = False
    if self.lora_rank > 0:
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
```

Only `lora_A` and `lora_B` are trained. LoRA $B$ is initialized to zero, so:

$$
y_{\text{init}} = (x V^\top) U^\top + b
\tag{4.8}
$$

which equals the SVD approximation.

## 4.8 The Full Algorithm

**Algorithm 1:** MinimalNOVA injection
```
Input:  model M, rank_cap r_max, energy η
Output: model M' with factorized attention

1:  for layer ℓ in transformer_layers(M) do
2:      src ← get_attention(ℓ)
3:      extract Q, K, V, O from src  (architecture-dependent)
4:
5:      for (W, target) in [(Q, q_proj), (K, k_proj), (V, v_proj), (O, out_proj)] do
6:          U, Σ, V ← svd(W)
7:          E(r) ← cumsum(Σ²) / sum(Σ²)
8:          r_eff ← min(r_max, |{i: E(i) < η}| + 1)
9:          U_r ← U[:, :r] · √Σ[:r]
10:         V_r ← √Σ[:r] · V[:r, :]
11:         target.down.weight[:r_eff, :] ← V_r
12:         target.up.weight[:, :r_eff] ← U_r
13:     end for
14:
15:     freeze_all_base_factors(ℓ)
16:     replace_attention(ℓ, new)
17: end for
18: return M'
```

## 4.9 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Weight extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per projection | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

For DistilBERT ($N_l = 6$, $d = 768$): ~10 seconds on RTX 4070.

## 4.10 Theoretical Notes

### 4.10.1 Optimality per Rank

Given rank $r$, the truncated SVD is optimal in Frobenius norm by Eckart–Young.

### 4.10.2 Energy Criterion

**Proposition.** The energy-adaptive rank satisfies:

$$
\frac{\|W - U_r V_r\|_F^2}{\|W\|_F^2} = 1 - E(r)
\tag{4.9}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = \|W\|_F^2 - \sum_{i \leq r} \sigma_i^2$. 
Dividing by $\|W\|_F^2$ gives (4.9). $\square$

This means: choosing $r$ with $E(r) \geq \eta$ guarantees relative 
reconstruction error $\leq \sqrt{1 - \eta}$.

For $\eta = 0.95$: error $\leq \sqrt{0.05} \approx 0.224$.
For $\eta = 0.99$: error $\leq \sqrt{0.01} = 0.10$.

### 4.10.3 Why Symmetric Distribution Helps

Distributing $\Sigma^{1/2}$ symmetrically ensures:
- Both factors have unit spectral norm (up to constants)
- Gradient magnitudes are balanced during LoRA training
- Numerical stability in FP16 (no factor dominates)

## 4.11 Summary

MinimalNOVA:
1. Factorizes each attention projection $W \to (U, V)$.
2. Selects rank via spectral energy $\eta$.
3. Uses LoRA for recovery.
4. Handles four attention architectures.
5. Achieves **real parameter reduction** of $d^2 / (2rd)$ per projection.

---