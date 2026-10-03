<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

MinimalSVD replaces the four attention projections of a transformer layer 
with **SVD-factorized linear modules**:

$$
W \to (U, V), \qquad
y = (x V^\top) U^\top + \text{LoRA}(x)
\tag{4.1}
$$

The factorization is done independently per projection. Rank is chosen 
adaptively per layer based on spectral energy.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input weight W ∈ R^{d_out × d_in}           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  SVD: W = U Σ V^T                             │
   │  Compute cumulative energy E(r)              │
   │  r_eff = min(r_max, argmin E(r) ≥ η)         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Truncate and symmetrize                      │
   │  U_r = U[:, :r] · √Σ[:r]                     │
   │  V_r = √Σ[:r] · V[:r, :]^T                   │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Store in FactorisedLinear                    │
   │  Freeze U, V; add trainable LoRA              │
   └──────────────────────────────────────────────┘
```

## 4.2 The `FactorisedLinear` Module

### 4.2.1 Forward Pass

The module computes:

$$
y = \text{up}(\text{down}(x)) + \text{LoRA}(x)
\tag{4.2}
$$

where:

- `down: nn.Linear(d_in, r, bias=False)` — stores $V_r \in \mathbb{R}^{r \times d_\text{in}}$
- `up: nn.Linear(r, d_out, bias=True)` — stores $U_r \in \mathbb{R}^{d_\text{out} \times r}$
- `LoRA(x) = (x A^\top) B^\top \cdot s` — optional adapter

### 4.2.2 Parameter Count

| Configuration | Parameters |
|---|---|
| Original `nn.Linear` | $d_\text{in} d_\text{out}$ |
| `FactorisedLinear` at rank $r$ | $r(d_\text{in} + d_\text{out})$ |
| `FactorisedLinear` + LoRA($r_L$) | $r(d_\text{in} + d_\text{out}) + r_L(d_\text{in} + d_\text{out})$ |

**Example** for a $768 \times 768$ projection:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| r=64 | 98,304 | **6.00×** |
| r=128 | 196,608 | 3.00× |
| r=64 + LoRA r=16 | 122,880 | 4.80× |

## 4.3 Spectral Energy-Adaptive Rank Selection

### 4.3.1 Algorithm

Given weight $W$ and hyperparameters $r_\text{max}$ (cap) and $\eta$ (energy):

1. Compute SVD: $W = U \Sigma V^\top$.
2. Compute cumulative energy:
   $$
   E(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_i \sigma_i^2}
   \tag{4.3}
   $$
3. Find the smallest $r$ with $E(r) \geq \eta$:
   $$
   r_\text{energy} = \left|\{i : E(i) < \eta\}\right| + 1
   \tag{4.4}
   $$
4. Cap the rank:
   $$
   r_\text{eff} = \min(r_\text{max}, r_\text{energy}, \min(d_\text{in}, d_\text{out}))
   \tag{4.5}
   $$

### 4.3.2 Why Energy?

The energy criterion captures **how concentrated the spectrum is**:

- **Rank-1 matrix** ($\sigma_{i>1} = 0$): $r_\text{energy} = 1$ for any 
  $\eta > 0$. Very compressible.
- **Identity matrix** ($\sigma_i = 1$): $r_\text{energy} = \lceil \eta d \rceil$. 
  Less compressible.

Real transformer weights lie between these extremes, with power-law or 
exponential decay.

### 4.3.3 Comparison to Fixed Rank

| Criterion | Pros | Cons |
|---|---|---|
| Fixed rank | Simple | Ignores heterogeneity |
| Energy | Adaptive | No task-specific calibration |
| Fisher-weighted | Task-aware | Requires calibration |

MinimalSVD uses energy: **training-free** and applicable wherever the model 
weights are available.

## 4.4 Weight Factorization

### 4.4.1 Symmetric Distribution

We distribute $\Sigma^{1/2}$ between the two factors:

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \qquad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{4.6}
$$

This gives $W_r = U_r V_r$ with balanced magnitudes, improving training 
stability.

### 4.4.2 Storage

The framework stores $V_r$ in `down.weight` and $U_r$ in `up.weight`:

```python
self.down.weight.data[:r, :] = V_r    # (r, d_in)
self.up.weight.data[:, :r] = U_r      # (d_out, r)
```

For $r < \text{target rank}$, remaining rows/columns are zero.

## 4.5 Weight Transfer from Pretrained Attention

MinimalSVD extracts $W_Q, W_K, W_V, W_O$ from the pretrained attention. This 
is nontrivial because different architectures store them differently.

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
\tag{4.7}
$$

(Conv1D stores this transposed.) MinimalSVD splits:

```python
w = src.c_attn.weight.data.clone()
if is_conv1d:
    w = w.t()   # (3d, d)
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 4.5.3 RoBERTa Nested

The output projection $W_O$ is at `attention.output.dense`, **outside** the 
inner attention. When replacing the inner attention, the outer `output.dense` 
remains in the model and applies the output projection **after** our module 
returns. So we set `is_nested_encoder = True` and skip `out_proj` in forward.

### 4.5.4 GQA (Qwen)

For GQA, $H_{kv} < H$. MinimalSVD factorizes each at its natural dimension 
and applies `repeat_kv` at forward.

## 4.6 Validation

### 4.6.1 Frobenius Norm Validation

Extracted weights are validated:

```python
for proj in ["q", "k", "v"]:
    w = extracted[proj][0]
    if w is not None:
        norm = torch.linalg.norm(w.float())
        if norm < 1e-4:
            raise RuntimeError(f"Near-zero weight on {proj}")
```

This catches catastrophic extraction failures.

### 4.6.2 Missing Weight Check

```python
missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
if not self.is_nested_encoder and extracted["o"][0] is None:
    missing.append("o")
if missing:
    raise RuntimeError(f"Missing required weights: {missing}")
```

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

Only `lora_A` and `lora_B` are trained. LoRA $B$ starts at zero, so:

$$
y_{\text{init}} = (x V^\top) U^\top + b
\tag{4.8}
$$

which equals the SVD approximation.

## 4.8 The Full Algorithm

**Algorithm 1:** MinimalSVD injection (single-layer)
```
Input:  layer ℓ, config c, energy η, rank cap r_max
Output: layer with factorized attention

1:  src ← get_attention(layer)
2:  if src is already MinimalSVD: return None
3:
4:  new_attn ← MinimalSVDAttention(c)
5:  try:
6:      new_attn.copy_weights(src)
7:  except Exception:
8:      return None
9:
10: # Set nested flag
11: if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
12:     new_attn.is_nested_encoder = True
13:     layer.attention.self = new_attn
14: else:
15:     new_attn.is_nested_encoder = False
16:     if hasattr(layer, "attention"):
17:         layer.attention = new_attn
18:     elif hasattr(layer, "self"):
19:         new_attn.is_nested_encoder = True
20:         layer.self = new_attn
21: return new_attn
```

**Whole-model traversal** is delegated to the caller:

```python
for layer in model.transformer.layer:
    inject_minimal_svd(layer, model.config, energy=0.95)
```

## 4.9 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per projection | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Truncation | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

For DistilBERT: ~10 seconds on RTX 4070.

## 4.10 Theoretical Notes

### 4.10.1 Optimality per Rank

Given rank $r$, the truncated SVD is optimal in Frobenius norm by 
Eckart–Young.

### 4.10.2 Energy Error Bound

**Proposition.** If $r_\text{eff}$ is the smallest rank such that 
$E(r_\text{eff}) \geq \eta$, then:

$$
\frac{\|W - W_{r_\text{eff}}\|_F}{\|W\|_F} \leq \sqrt{1 - \eta}
\tag{4.9}
$$

**Proof.** $\|W - W_r\|_F^2 = \|W\|_F^2(1 - E(r))$. Since 
$E(r_\text{eff}) \geq \eta$, the bound follows. $\square$

**Corollary.** For $\eta = 0.95$: relative error $\leq \sqrt{0.05} \approx 0.224$. 
For $\eta = 0.99$: relative error $\leq 0.10$.

### 4.10.3 Symmetric Distribution Stability

Distributing $\Sigma^{1/2}$ symmetrically ensures both factors have similar 
spectral norms, improving training stability in FP16.

## 4.11 Summary

MinimalSVD:
1. Factorizes each attention projection $W \to (U, V)$.
2. Selects rank via spectral energy $\eta$.
3. Uses LoRA for recovery.
4. Handles four attention architectures.
5. Achieves **real parameter reduction** of $d^2 / (r(d_\text{in} + d_\text{out}))$.

---