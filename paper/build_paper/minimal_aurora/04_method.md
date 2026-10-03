# 4. Method

## 4.1 Overview

MinimalAURORA replaces each attention projection with a factorized 
approximation plus LoRA residual. **Figure 4.1** shows the pipeline.

```
    ┌────────────────────────────────────────┐
    │  Original weight W ∈ R^{m×n}           │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 1: SVD of W                      │
    │  U, Σ, V^T = svd(W)                    │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 2: Energy-based rank selection   │
    │  r = min{r : E(r) ≥ ε}                 │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 3: Factorize                     │
    │  down.weight = Σ^{1/2} V^T             │
    │  up.weight   = U Σ^{1/2}               │
    └────────────────┬───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  Step 4: Add LoRA (B = 0)              │
    │  Freeze SVD factors                    │
    └────────────────────────────────────────┘
```

## 4.2 Energy-Based Rank Selection

### 4.2.1 Rank Function

For each projection $W$:

$$
U, \Sigma, V^\top = \text{SVD}(W)
\tag{4.1}
$$

Compute cumulative energy:

$$
E(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}
\tag{4.2}
$$

Select minimal rank:

$$
r^{(\varepsilon)} = \min\{r : E(r) \geq \varepsilon\}
\tag{4.3}
$$

### 4.2.2 The `decompose_from_weight` Method

```python
def decompose_from_weight(self, W, bias=None, energy=None):
    U, S, Vt = torch.linalg.svd(W)
    target_energy = energy if energy is not None else self.energy
    cum_energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
    calc_rank = int((cum_energy < target_energy).sum().item()) + 1
    r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
    r = max(1, r)
    
    sqrt_s = torch.sqrt(S[:r])
    A = U[:, :r] * sqrt_s
    B = sqrt_s[:, None] * Vt[:r, :]
    
    self.up.weight.data[:, :r] = A
    self.down.weight.data[:r, :] = B
    self.freeze_base_weights()
```

## 4.3 Forward Pass

$$
y = \text{up}(\text{down}(x)) + \frac{\alpha}{r_L} B A x
\tag{4.4}
$$

where:
- `down(x) = x V_r^\top` (down-projection, no bias)
- `up(z) = z U_r^\top + b` (up-projection, with bias)
- `B A x` (LoRA branch)

## 4.4 Multi-Architecture Weight Transfer

The `copy_weights` routine handles four cases:

### 4.4.1 GPT-2 (Fused Conv1D)

```python
w = src.c_attn.weight.data.t()  # transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
# Also c_proj
```

### 4.4.2 RoBERTa / BERT (Nested)

```python
s = src.self
q_w = s.query.weight.data
k_w = s.key.weight.data
v_w = s.value.weight.data
o_w = src.output.dense.weight.data
```

### 4.4.3 DistilBERT (Flat)

```python
q_w = src.q_lin.weight.data
k_w = src.k_lin.weight.data
v_w = src.v_lin.weight.data
o_w = src.out_lin.weight.data
```

### 4.4.4 Qwen / LLaMA

```python
q_w = src.q_proj.weight.data
k_w = src.k_proj.weight.data
v_w = src.v_proj.weight.data
o_w = src.o_proj.weight.data
```

### 4.4.5 GQA Handling

For GQA where $H_{kv} < H$, the extracted K/V weights may have different 
dimensions. `decompose_from_weight` handles this:

```python
if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
    rep = self.out_features // W.shape[0]
    W = W.repeat(rep, 1)
elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
    rep = W.shape[0] // self.out_features
    W = W.view(self.out_features, rep, -1).mean(dim=1)
```

## 4.5 Parameter Count

| Component | Params |
|---|---|
| Down | $r \cdot d_\text{in}$ |
| Up | $r \cdot d_\text{out} + d_\text{out}$ (bias) |
| LoRA | $r_L(d_\text{in} + d_\text{out})$ |
| **Total** | $(r + r_L)(d_\text{in} + d_\text{out}) + d_\text{out}$ |

**Baseline:** $d_\text{in} d_\text{out}$.

For $d = 768$, $r = 32$, $r_L = 16$:

| Config | Params | Ratio |
|---|---|---|
| Baseline | 589,824 | 1.00× |
| MinimalAURORA | 73,728 + 768 = 74,496 | **7.9×** |

## 4.6 Full Algorithm

**Algorithm 1:** MinimalAURORA factorization
```
Input:  W ∈ R^{m×n}, ε, r_max, r_L
Output: FactorisedLinear with frozen base + trainable LoRA

1:  U, Σ, V^T ← svd(W)
2:  cum_energy ← cumsum(Σ²) / sum(Σ²)
3:  r ← min{r : cum_energy[r] ≥ ε}
4:  r ← min(r, r_max, m, n)
5:  sqrt_Σ ← sqrt(Σ[:r])
6:  up.weight ← U[:, :r] · sqrt_Σ
7:  down.weight ← sqrt_Σ · V^T[:r, :]
8:  freeze(up.weight); freeze(down.weight)
9:  initialize lora_A (Kaiming), lora_B = 0
10: return
```

## 4.7 Complexity Analysis

| Operation | Time |
|---|---|
| SVD per projection | $\mathcal{O}(mn \min(m,n))$ |
| Rank selection | $\mathcal{O}(\min(m,n))$ |
| **Per layer total** | $\mathcal{O}(mn \min(m,n))$ |

For DistilBERT with 4 projections per layer, 6 layers: ~30 SVDs.

**Total compression time:** ~2.3 s seconds.

## 4.8 Theoretical Notes

### 4.8.1 Energy Bound

**Claim.** With energy threshold $\varepsilon$:

$$
\|W - W_{r(\varepsilon)}\|_F \leq \sqrt{1 - \varepsilon} \cdot \|W\|_F
\tag{4.5}
$$

**Proof.** $\|W - W_r\|_F^2 = \sum_{i > r} \sigma_i^2 = (1 - E(r)) \|W\|_F^2$. 
With $E(r) \geq \varepsilon$, the bound follows. $\square$

### 4.8.2 Causality

**Proof.** Position-wise projections + causal attention mask. $\square$

### 4.8.3 Nested Encoder Correctness

**Claim.** For nested encoders (RoBERTa/BERT), the output projection is 
**not applied** internally.

**Proof.** RoBERTa wraps attention as `BertAttention` with `self` (Q/K/V) 
and `output.dense` (output projection). MinimalAURORA detects this via 
`is_nested_encoder = True` and returns the raw context, letting the outer 
layer apply the projection. $\square$

---