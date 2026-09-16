# 4. Method

## 4.1 Overview

HGRN2-GatedDeltaNet replaces standard attention with a **two-branch linear 
recurrent module**. Both branches process the sequence sequentially with a 
fixed-size state, giving $\mathcal{O}(L)$ complexity. A learned gate fuses 
the two branches per token.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
       ┌──────┴──────┐
       ▼             ▼
    ┌──────┐    ┌──────────┐
    │ HGRN │    │ DeltaNet │
    │ -2   │    │          │
    └──┬───┘    └─────┬────┘
       │              │
       │  γ = σ(W_g X)│
       │     │        │
       ▼     ▼        ▼
       ┌───────────────────┐
       │ Y = γ⊙Y₁ + (1-γ)⊙Y₂│
       └─────────┬─────────┘
                 │
                 ▼
              out_proj
                 │
                 ▼
              Output Y
```

## 4.2 HGRN-2 Branch

The HGRN-2 branch (Qin et al., 2024) computes:

$$
u_t = \text{SiLU}(W_u x_t + b_u)
\tag{4.1}
$$

$$
\alpha_t = \sigma(W_\alpha x_t + b_\alpha)
\tag{4.2}
$$

$$
s_t = \alpha_t \odot s_{t-1} + (1 - \alpha_t) \odot u_t
\tag{4.3}
$$

The output is $y_t^\text{HGRN} = s_t$ (or optionally projected).

**Initialization.** $s_0 = 0$. Gates $W_\alpha$ are initialized so that 
$\alpha_t \approx 0.5$ initially.

## 4.3 Gated DeltaNet Branch

The Gated DeltaNet branch maintains a matrix state per head:

$$
S_t^h = \alpha_t S_{t-1}^h + \beta_t (v_t^h - S_{t-1}^h k_t^h) (k_t^h)^\top
\tag{4.4}
$$

where:

- $k_t^h = W_k x_t / \|W_k x_t\|$ (normalized key).
- $v_t^h = W_v x_t$ (value).
- $\alpha_t, \beta_t \in (0, 1)$ are scalar gates.

The output is:

$$
y_t^{h, \text{Delta}} = S_t^h q_t^h
\tag{4.5}
$$

where $q_t^h = W_q x_t$.

**Why the Delta rule?** The update $\beta_t (v_t - S_{t-1} k_t)$ writes only 
the **error** into memory, avoiding redundant writes.

## 4.4 Fusion Gate

The fusion gate is a linear projection:

$$
\gamma_t = \sigma(W_g x_t + b_g)
\tag{4.6}
$$

with $W_g \in \mathbb{R}^{d \times d}$ and $b_g \in \mathbb{R}^d$.

**Initialization.** We set $W_g = 0$ and $b_g = 0$, giving 
$\gamma_t = 0.5$ initially (equal blend). During training, the gate learns 
to prefer the branch that best fits each token.

**Why the gate matters.** At initialization, both branches are 
approximately equally weighted. If one branch's `copy_weights` fails, we 
bias the gate toward the surviving branch (see §4.6).

## 4.5 Combined Output

The final output is:

$$
Y = \gamma \odot Y^\text{HGRN} + (1 - \gamma) \odot Y^\text{Delta}
\tag{4.7}
$$

followed by an output projection:

$$
\text{out} = W_o Y + b_o
\tag{4.8}
$$

## 4.6 Failure Handling

### 4.6.1 Both Branches Fail

If both `copy_weights` calls fail, we **abort** with `RuntimeError`. This 
prevents injecting an untrained random module.

### 4.6.2 One Branch Fails

If only HGRN succeeds:
```python
self.gate.weight.zero_()
self.gate.bias.fill_(4.0)  # sigmoid(4) ≈ 0.98
```

If only DeltaNet succeeds:
```python
self.gate.weight.zero_()
self.gate.bias.fill_(-4.0)  # sigmoid(-4) ≈ 0.02
```

This ensures the gate leans toward the surviving branch.

### 4.6.3 Output Projection Fails

If `out_proj` weights cannot be found, we **abort** with `RuntimeError`. 
This prevents injecting a random output projection.

## 4.7 Algorithm

**Algorithm 1:** HGRN2-GatedDeltaNet forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: Y_hgrn ← HGRN2Branch(X)                 ▷ Sequential O(L)
 2: Y_delta ← DeltaNetBranch(X)             ▷ Sequential O(L)
 3: γ ← σ(W_g X + b_g)                      ▷ (B, L, d)
 4: Y ← γ ⊙ Y_hgrn + (1-γ) ⊙ Y_delta
 5: Y ← out_proj(Y)
 6: return Y
```

## 4.8 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| HGRN-2 | $\mathcal{O}(L d)$ | $\mathcal{O}(d)$ |
| DeltaNet | $\mathcal{O}(L H d^2)$ | $\mathcal{O}(H d^2)$ |
| Gate | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d)$ |
| out_proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(d^2)$ |
| **Total** | $\mathcal{O}(L H d^2)$ | $\mathcal{O}(H d^2)$ |

Compare to standard attention: $\mathcal{O}(L^2 H d)$. Our method is 
**linear** in $L$.

## 4.9 Parameter Analysis

**HGRN2-GatedDeltaNet does NOT reduce parameter count.**

| Component | Baseline | Ours |
|---|---|---|
| Q projection | $d^2$ | $d^2$ (frozen) |
| K projection | $d^2$ | $d^2$ (frozen) |
| V projection | $d^2$ | $d^2$ (frozen) |
| O projection | $d^2$ | $d^2$ (frozen) |
| HGRN-2 params | 0 | $\sim 4d^2$ |
| DeltaNet params | 0 | $\sim 4d^2$ |
| Gate | 0 | $d^2$ |
| LoRA | 0 | $8 r_L d$ |

**Total increase:** ~9$d^2$ per layer.

For $d = 768$:

| Config | Params per layer |
|---|---|
| Baseline | 2.36M |
| HGRN2-GatedDeltaNet | ~7.7M |
| **Increase** | **+226%** |

This is a significant parameter increase. We report it honestly.

## 4.10 Training

We train only:
- The gate $W_g, b_g$.
- LoRA adapters on Q, K, V, O, and the branch-specific projections.

All base weights from the pretrained attention and the two branches are 
frozen.

**Trainable params:** for $d = 768$, $r_L = 16$:
- LoRA: $8 \cdot 16 \cdot 768 = 98,304$
- Gate: $768^2 + 768 = 590,592$
- **Total:** ~689K per layer

For 12 layers: ~8.3M trainable.

---