# 4. Method

## 4.1 Overview

Mamba-Inspired Gated SSM Attention replaces standard attention with a 
**two-branch gated state-space formulation**:

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ in_proj: d → 2d      │
    │   first d = p1       │  ← SSM branch
    │   second d = p2      │  ← Gate branch
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Learned decay        │
    │   γ_i = σ(-exp(A_i)) │
    │   softmax over dist  │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ p1' = decay(p1)      │
    │ out = SiLU(p1') * p2 │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ out_proj: d → d      │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Preservation gate    │
    │ α·out + (1-α)·X      │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 Two-Branch In-Projection

Instead of standard Q, K, V projections, we use:

$$
[p_1; p_2] = W_\text{in} X + b_\text{in}
\tag{4.1}
$$

where $W_\text{in} \in \mathbb{R}^{d \times 2d}$ and $p_1, p_2 \in \mathbb{R}^{B \times L \times d}$.

**Weight reuse.**

- $p_1$ branch: copied from $W_V$ (or $W_Q$ if V unavailable)
- $p_2$ branch: copied from $W_Q$ (or $W_K$ if Q unavailable)

This gives:

$$
W_\text{in} = \begin{pmatrix} W_V \\ W_Q \end{pmatrix}
\tag{4.2}
$$

### 4.2.1 Padding for Different Head Counts

If $d_\text{out}$ of the source projection is smaller than $d$, we **repeat**:

```python
if w_main.shape[0] < D and D % w_main.shape[0] == 0:
    rep = D // w_main.shape[0]
    w_main = w_main.repeat(rep, 1)
```

## 4.3 Learned Decay

### 4.3.1 Decay Parameterization

For each dimension $i \in \{1, \dots, d\}$:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{4.3}
$$

We initialize $A_{\log, i} = \log(i)$ (from `torch.log(arange(1, D+1))`), so:

$$
\gamma_i = \sigma(-i)
\tag{4.4}
$$

For small $i$, $\gamma_i \approx \sigma(-1) \approx 0.27$; for large $i$, 
$\gamma_i \to 0$.

### 4.3.2 Causality

For decoders, we define:

$$
\text{dist}_{ij} = \begin{cases} j - i & j \geq i \\ 0 & j < i \end{cases}
\tag{4.5}
$$

**Wait, this is reversed.** Let me re-read the code:

```python
dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0)  # [L, L]
# pos.unsqueeze(1) is i, pos.unsqueeze(0) is j
# dist[i, j] = max(0, i - j)
```

So $\text{dist}_{ij} = \max(0, i - j)$ — the distance from position $j$ to 
position $i$ **for $j \leq i$**.

The log weights:

$$
\log w_{ij}^{(d)} = \text{dist}_{ij} \cdot \log \gamma_d
\tag{4.6}
$$

Then softmax over $j$ (dim=1):

$$
w_{ij}^{(d)} = \frac{\exp(\log w_{ij}^{(d)})}{\sum_{j'} \exp(\log w_{ij'}^{(d)})}
\tag{4.7}
$$

**Interpretation.** For query at position $i$, key at position $j$:

- If $j = i$: $\text{dist}_{ij} = 0$, so $\log w = 0$, giving $w = 1/Z$ 
  (highest after normalization).
- If $j = i - 1$: $\text{dist}_{ij} = 1$, so $\log w = \log \gamma_d$.
- If $j \to 0$: $\text{dist}_{ij} = i$, so $\log w = i \log \gamma_d$ 
  (smallest).

**Effect.** Recent keys get more weight; older keys get exponentially 
decayed weight. This emulates SSM's "forget gate".

### 4.3.3 Causal Mask

The causal mask sets entries with $j > i$ to $-\infty$:

```python
causal_mask = torch.tril(torch.ones(L, L))
log_weights = log_weights.masked_fill(~causal_mask, float("-inf"))
```

## 4.4 Bidirectional Mode

For encoders, we compute forward and backward distances:

$$
\text{dist}^\text{fwd}_{ij} = \max(0, i - j), \quad
\text{dist}^\text{bwd}_{ij} = \max(0, j - i)
\tag{4.8}
$$

Both are masked appropriately:

- Forward: $j \leq i$ (lower triangular)
- Backward: $j \geq i$ (upper triangular)

Then:

$$
\text{Out} = 0.5 \cdot (w^\text{fwd} p_1 + w^\text{bwd} p_1)
\tag{4.9}
$$

## 4.5 Gated Fusion

$$
\text{Out}_\text{SSM} = \text{SiLU}(p_1^\text{decayed}) \odot p_2
\tag{4.10}
$$

where SiLU is:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
\tag{4.11}
$$

## 4.6 Output Projection

Standard LoRALinear:

$$
\text{Out} = W_O \cdot \text{Out}_\text{SSM} + b_O
\tag{4.12}
$$

with $W_O$ copied from the pretrained output projection.

## 4.7 Preservation Gate

A learnable scalar $g$ initialized to 2.0:

$$
\alpha = \sigma(g) \approx 0.88
\tag{4.13}
$$

Final output:

$$
\text{Final} = \alpha \cdot \text{Out} + (1 - \alpha) \cdot X
\tag{4.14}
$$

**Why this helps.** At the start of training, the SSM output may be noisy. 
The preservation gate keeps the residual input as a fallback. As training 
progresses, $\alpha$ can shift toward 1.0 if the SSM output improves.

## 4.8 Full Algorithm

**Algorithm 1:** Mamba-Inspired Gated SSM Attention forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, is_decoder
Output: Y ∈ R^{B×L×d}

 1: X ← RoPE(X) if position_embeddings provided
 2: proj ← in_proj(X)                        ▷ [B, L, 2d]
 3: p1, p2 ← chunk(proj, 2, dim=-1)          ▷ Two branches
 4: γ ← σ(-exp(A_log[:d]))                    ▷ [d]
 5: if is_decoder and L > 1:
 6:     dist ← max(0, i - j)                  ▷ [L, L]
 7:     log_w ← dist ⊗ log γ                  ▷ [L, L, d]
 8:     log_w ← mask_causal(log_w)
 9:     w ← softmax(log_w, dim=1)             ▷ [L, L, d]
10:     p1 ← einsum("ijd,bjd->bid", w, p1)
11:     gated ← SiLU(p1) * p2
12: elif L > 1:
13:     # Bidirectional
14:     dist_fwd ← max(0, i - j); dist_bwd ← max(0, j - i)
15:     w_fwd ← softmax(dist_fwd ⊗ log γ, dim=1)
16:     w_bwd ← softmax(dist_bwd ⊗ log γ, dim=1)
17:     p1_mix ← 0.5 * (w_fwd @ p1 + w_bwd @ p1)
18:     gated ← SiLU(p1_mix) * p2
19: else:
20:     gated ← SiLU(p1) * p2
21: Out ← out_proj(gated)                       ▷ d → d
22: Out ← mask_padding(Out, attention_mask)
23: α ← σ(pres_gate)
24: Final ← α * Out + (1 - α) * X
25: return Final
```

## 4.9 Complexity Analysis

### 4.9.1 Time Complexity

| Operation | Complexity |
|---|---|
| in_proj | $\mathcal{O}(L d^2)$ |
| Decay weights | $\mathcal{O}(L^2 d)$ |
| Softmax | $\mathcal{O}(L^2 d)$ |
| Weighted sum | $\mathcal{O}(L^2 d)$ |
| out_proj | $\mathcal{O}(L d^2)$ |

**Total:** $\mathcal{O}(L^2 d + L d^2)$.

For $L \gg d$: **$\mathcal{O}(L^2 d)$** — quadratic in $L$.

### 4.9.2 Memory Complexity

The weight tensor is $[L, L, d]$, so:

$$
\text{Memory} = \mathcal{O}(L^2 d)
\tag{4.15}
$$

**This is NOT O(L).**

### 4.9.3 Comparison with True Mamba

| Property | True Mamba | Our Implementation |
|---|---|---|
| Time | $\mathcal{O}(L)$ | $\mathcal{O}(L^2 d)$ |
| Memory | $\mathcal{O}(L d)$ | $\mathcal{O}(L^2 d)$ |
| Selective | Input-dependent | Learned per-dim |
| Scan | Parallel scan | Softmax over distance |

**The docstring claim of "O(L)" is incorrect.**

## 4.10 Parameter Analysis

| Component | Params |
|---|---|
| in_proj | $2 d^2$ |
| out_proj | $d^2$ |
| A_log | $d$ |
| pres_gate | 1 |
| LoRA (both proj) | $r_L (2d + d) \cdot 2$ |

For $d = 768$, $r_L = 16$:

| Component | Params |
|---|---|
| in_proj | 1,179,648 |
| out_proj | 589,824 |
| A_log | 768 |
| pres_gate | 1 |
| LoRA | ~66K |
| **Total** | **~1.84M** |

vs baseline attention ($4 d^2 = 2.36M$):

**Reduction: ~22%** at $r_L = 16$.

## 4.11 Theoretical Notes

### 4.11.1 Decay Interpretation

The learned decay $\gamma_i$ controls the SSM's "forget rate" per dimension:

- $\gamma_i \approx 1$: no forgetting → attends to all history
- $\gamma_i \approx 0$: rapid forgetting → attends only to recent

The init $A_{\log, i} = \log(i)$ gives $\gamma_i = \sigma(-i)$, so higher 
dimensions forget faster.

### 4.11.2 Softmax Over Distance

The softmax over distance is equivalent to a **geometric decay prior**:

$$
w_{ij} \propto \gamma_i^{|i - j|}
\tag{4.16}
$$

This is the discrete analog of the SSM's exponential decay.

### 4.11.3 Preservation Gate

**Claim.** The preservation gate ensures the initial forward pass approximates 
the identity transformation.

**Proof.** At init, $\alpha = \sigma(2.0) \approx 0.88$. So the output is 
$0.88 \cdot \text{Out}_\text{SSM} + 0.12 \cdot X$. If the SSM output is 
approximately zero (as $B = 0$ for LoRA), the output is $\approx 0.12 X$. 
$\square$

### 4.11.4 Causality

**Claim.** For decoders, causality is preserved.

**Proof.** The causal mask sets $w_{ij} = 0$ for $j > i$ (upper triangular 
to $-\infty$ in log-space). After softmax, $w_{ij} = 0$ for $j > i$. So 
output at $i$ depends only on inputs $j \leq i$. $\square$

---