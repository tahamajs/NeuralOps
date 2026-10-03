<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Dobi-SVD replaces each target linear layer with a **differentiable 
factorized module**:

$$
y = (x (V \odot \sigma(\alpha))^\top) U^\top + b
\tag{4.1}
$$

where:

- $U \in \mathbb{R}^{d_\text{out} \times r}$ — left factor (frozen initially, initialized from SVD)
- $V \in \mathbb{R}^{r \times d_\text{in}}$ — right factor (frozen, gate-compensated)
- $\alpha \in \mathbb{R}^r$ — learnable gate logits (trainable)
- $b \in \mathbb{R}^{d_\text{out}}$ — bias

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input weight W ∈ R^{d_out × d_in}           │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  SVD: W = U Σ V^T                             │
   │  Rank r selected by energy threshold          │
   │  U_r = U[:, :r] Σ[:r]^1/2                     │
   │  V_r = Σ[:r]^1/2 V[:r, :]^T                   │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Gate compensation:                           │
   │  V_r ← V_r / σ(α_init)                        │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Create DobiSVDLinearLayer:                   │
   │  U ← U_r                                       │
   │  V ← V_r                                       │
   │  α ← init_gate_val (learnable)                │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Recovery training:                            │
   │  Train α and LoRA adapters                    │
   └──────────────────────────────────────────────┘
```

## 4.2 The `DobiSVDLinearLayer`

### 4.2.1 Module Structure

```python
class DobiSVDLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True,
                 init_gate_val=2.5, ...):
        self.V = nn.Parameter(torch.empty(rank, in_features))
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.gate = nn.Parameter(torch.full((rank,), fill_value=init_gate_val))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
```

### 4.2.2 Forward Pass

```python
def forward(self, x):
    g = torch.sigmoid(self.gate).unsqueeze(1)  # (rank, 1)
    V_gated = self.V * g  # (rank, in_features)
    latent = F.linear(x, V_gated)  # (..., rank)
    out = F.linear(latent, self.U, self.bias)  # (..., out_features)
    return out
```

### 4.2.3 Parameter Count

| Configuration | Parameters |
|---|---|
| Original `nn.Linear` | $d_\text{in} d_\text{out}$ |
| Dobi-SVD at rank $r$ | $r d_\text{in} + d_\text{out} r + r$ |
| = | $r(d_\text{in} + d_\text{out} + 1)$ |

**Example** for a $768 \times 768$ projection at $r = 128$:

| Config | Params | Ratio |
|---|---|---|
| Original | 589,824 | 1.00× |
| Dobi-SVD $r=128$ | 196,736 | 3.00× |
| Dobi-SVD $r=64$ | 98,368 | 6.00× |

## 4.3 Initialization Procedure

### 4.3.1 Step 1: SVD

$$
W = U \Sigma V^\top
\tag{4.2}
$$

### 4.3.2 Step 2: Truncate at rank $r$

$$
U_r = U_{:, :r} \Sigma_{:r, :r}^{1/2}, \quad
V_r = \Sigma_{:r, :r}^{1/2} V_{:r, :}^\top
\tag{4.3}
$$

### 4.3.3 Step 3: Gate compensation

Let $\sigma_\text{init} = \sigma(\alpha_\text{init})$. Pre-scale $V_r$:

$$
V_r \leftarrow V_r / \sigma_\text{init}
\tag{4.4}
$$

This ensures the initial forward equals the SVD reconstruction.

### 4.3.4 Step 4: Store

```python
layer.U.data.copy_(U_r)
layer.V.data.copy_(V_r)
layer.gate.data.fill_(init_gate_val)
```

## 4.4 Rank Selection

### 4.4.1 Parameter Reduction Constraint

The rank must satisfy:

$$
r (d_\text{in} + d_\text{out} + 1) < d_\text{in} d_\text{out}
\tag{4.5}
$$

For square matrices at $d = 768$:

$$
r < \frac{d^2}{2d + 1} \approx \frac{d}{2}
\tag{4.6}
$$

Dobi-SVD uses an upper bound:

$$
r_\text{max} = \max\!\left(1, \min\!\left(d_\text{in} - 1, d_\text{out} - 1, \left\lfloor \frac{d_\text{in} d_\text{out}}{d_\text{in} + d_\text{out}} \right\rfloor - 1, {{RANK\_CAP}}\right)\right)
\tag{4.7}
$$

Default `RANK_CAP = 128`.

### 4.4.2 Initial Rank

If `rank` is not specified, use:

$$
r_\text{init} = \max\!\left(1, \text{round}({{INIT\_RANK\_RATIO}} \cdot \min(d_\text{in}, d_\text{out}))\right)
\tag{4.8}
$$

Default `INIT_RANK_RATIO = 0.5`.

### 4.4.3 Adaptive Retry

If reconstruction error exceeds `max_error`, Dobi-SVD retries with:

$$
r_{k+1} = \min(1.5 \cdot r_k, r_\text{max})
\tag{4.9}
$$

up to `max_retries` (default 2).

## 4.5 Training

### 4.5.1 Trainable Parameters

- **Gates** $\alpha \in \mathbb{R}^r$ (one scalar per rank component)
- **LoRA adapters** (attached to the wrapped layer)
- **Non-target layers** are frozen

### 4.5.2 Loss

Standard cross-entropy on the downstream task:

$$
\mathcal{L} = -\frac{1}{N} \sum_i \log p_\theta(y_i \mid x_i)
\tag{4.10}
$$

Gradient flows through $\sigma(\alpha)$ to the gate logits.

## 4.6 Post-Hoc Rank Pruning

After training, gates are inspected. Let:

$$
r_\text{eff} = \left|\{i : \sigma(\alpha_i) > \tau_g\}\right|
\tag{4.11}
$$

for a threshold $\tau_g$ (e.g., 0.5). Components with $\sigma(\alpha_i) \leq \tau_g$ 
can be **removed** without retraining, at the cost of a small accuracy drop.

**Algorithm 2:** Post-hoc rank pruning
```
Input:  trained DobiSVDLinearLayer, threshold τ_g
Output: pruned layer with rank r_eff

1:  gates ← sigmoid(layer.gate)
2:  keep_idx ← where(gates > τ_g)
3:  r_eff ← len(keep_idx)
4:  V_pruned ← layer.V[keep_idx, :]
5:  U_pruned ← layer.U[:, keep_idx]
6:  gate_pruned ← layer.gate[keep_idx]
7:  return DobiSVDLinearLayer with (U_pruned, V_pruned, gate_pruned)
```

## 4.7 The Full Algorithm

**Algorithm 1:** Dobi-SVD compression
```
Input:  model M, rank r, gate_init α_0, threshold τ
Output: compressed model M'

1:  for each target layer ℓ in M do
2:      W ← weight(ℓ)
3:      U, Σ, V^T ← svd(W)
4:      U_r ← U[:, :r] · Σ[:r]^1/2
5:      V_r ← Σ[:r]^1/2 · V^T[:r, :]
6:      V_r ← V_r / σ(α_0)             ▷ Gate compensation
7:
8:      # Reconstruction validation
9:      W_recon ← U_r · (V_r · σ(α_0))
10:     err ← ‖W - W_recon‖_F / ‖W‖_F
11:     if err > τ:
12:         if fallback: skip ℓ
13:         else: escalate r and retry
14:
15:     # Replace
16:     layer_new ← DobiSVDLinearLayer(U_r, V_r, gate=α_0)
17:     replace(ℓ, layer_new)
18: end for
19:
20: # Recovery training (external)
21: train_gates_and_lora(M)
22:
23: # Post-hoc pruning (optional)
24: for each compressed layer:
25:     prune_gates_below(layer, τ_g)
26: return M
```

## 4.8 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Extraction | $\mathcal{O}(N_l d^2)$ | $\mathcal{O}(d^2)$ |
| SVD per layer | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ |
| Reconstruction | $\mathcal{O}(N_l r d^2)$ | $\mathcal{O}(d^2)$ |
| Total | $\mathcal{O}(N_l d^3)$ | $\mathcal{O}(d^2)$ |

## 4.9 Theoretical Notes

### 4.9.1 Lossless Initialization

**Proposition.** If gates are initialized at $\alpha_0$ and $V$ is 
pre-scaled by $1/\sigma(\alpha_0)$, then the initial forward pass equals 
the truncated SVD:

$$
y_0 = (x (V_r / \sigma(\alpha_0) \odot \sigma(\alpha_0))^\top) U_r^\top = (x V_r^\top) U_r^\top
\tag{4.12}
$$

$\square$

### 4.9.2 Gradient of Gate

The gradient of the loss with respect to $\alpha_i$:

$$
\frac{\partial \mathcal{L}}{\partial \alpha_i}
= \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial \sigma(\alpha_i)} \cdot \sigma(\alpha_i)(1 - \sigma(\alpha_i))
\tag{4.13}
$$

The $\sigma(1-\sigma)$ factor provides a natural gradient landscape 
that can push gates toward 0 (pruning) or 1 (retention).

### 4.9.3 Rank Reduction Bound

If after training we prune all gates with $\sigma(\alpha_i) \leq \tau_g$, the 
effective rank is:

$$
r_\text{eff} = \#\{i : \sigma(\alpha_i) > \tau_g\}
\tag{4.14}
$$

and the parameter count is:

$$
|\theta_\text{eff}| = r_\text{eff}(d_\text{in} + d_\text{out} + 1)
\tag{4.15}
$$

## 4.10 Summary

Dobi-SVD:
1. Factorizes each target layer with SVD.
2. Adds a learnable gate per rank component.
3. Pre-scales $V$ for lossless initialization.
4. Trains gates jointly with LoRA recovery.
5. Optionally prunes gates post-hoc.
6. Achieves **real parameter reduction** with **learnable rank**.

---