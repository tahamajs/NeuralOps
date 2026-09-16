# 4. Method

## 4.1 Overview

PADRe replaces softmax attention with a **polynomial kernel**. The forward 
pass:

1. **Project Q, K, V** via standard LoRALinear.
2. **Apply RoPE** if applicable.
3. **GQA expansion** of K, V.
4. **Score computation:** $s_{ij} = (q_i \cdot k_j)/\sqrt{d_h}$.
5. **Masking:** invalid positions set to large negative before centering.
6. **Max-centering:** subtract $\max_j s_{ij}$ from all scores.
7. **Polynomial activation:** $\text{attn} = (1 + s/d)^d$ with ReLU clamp.
8. **Masking again:** zero invalid positions.
9. **Normalize:** divide by the sum over $j$.
10. **Dropout.**
11. **Context:** $\text{attn} \cdot V$.

```
        Input X
           │
           ▼
    ┌──────────────────┐
    │ Q/K/V projection │
    │ + RoPE           │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ GQA expansion    │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Scores: QK^T/√d_h│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Apply masks      │
    │ (causal, padding)│
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Max-center       │
    │ s ← s - max(s)   │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Polynomial       │
    │ (1 + s/d)^d      │
    │ ReLU-clamp base  │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Zero-invalid     │
    │ Normalize        │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ Dropout          │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ attn @ V         │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │ out_proj         │
    └────────┬─────────┘
             │
        Output
```

## 4.2 Score Computation

Standard scaled dot-product:

$$
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_h}}
\tag{4.1}
$$

For GQA models, $K$ is expanded via `repeat_kv` before this step.

## 4.3 Masking

We combine two masks:

1. **Padding mask:** $k_j$ is a padding token.
2. **Causal mask** (decoders only): $j > i$.

Both are combined into a single boolean `invalid_mask`:

$$
\text{invalid}_{ij} = \text{pad}(j) \lor \text{causal}(i, j)
\tag{4.2}
$$

## 4.4 Max-Centering

For each query $i$, find the maximum score over **valid** keys:

$$
m_i = \max_{j: \text{valid}_{ij}} s_{ij}
\tag{4.3}
$$

If all keys are invalid (rare), we set $m_i = 0$ to avoid NaN:

```python
thresh = -5e3 if scores.dtype in (fp16, bf16) else -1e8
max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
```

Then center:

$$
s_{ij} \leftarrow (s_{ij} - m_i) / T
\tag{4.4}
$$

where $T$ is a temperature.

## 4.5 Polynomial Activation

Apply the polynomial kernel:

$$
\text{base}_{ij} = \text{ReLU}\!\left(1 + \frac{s_{ij}}{d}\right)
\tag{4.5}
$$

$$
\text{attn}_{ij} = \text{base}_{ij}^d
\tag{4.6}
$$

**Why ReLU?** If $s_{ij} < -d$ (very negative), the base is negative. Raising 
a negative number to a non-integer power gives NaN. The ReLU prevents this 
and also creates natural sparsity (positions with $s_{ij} < -d$ get zero weight).

**Why $1/d$ scaling?** It keeps the base close to 1 for moderate scores:

$$
\left(1 + \frac{s}{d}\right)^d \approx e^s \quad \text{for large } d
\tag{4.7}
$$

So PADRe is a **first-order approximation of the softmax kernel**.

## 4.6 Zero Invalid Positions

After the polynomial activation, we zero invalid positions:

$$
\text{attn}_{ij} \leftarrow 0 \quad \text{if invalid}_{ij}
\tag{4.8}
$$

**Why after, not before?** Because masking before centering would put 
$-\infty$ into the max, which produces $-\infty$ everywhere. We mask 
before centering (using a large finite value like $-10^4$) for the max, 
then zero after the polynomial for the final normalization.

## 4.7 Normalization

$$
\text{attn}_{ij} \leftarrow \frac{\text{attn}_{ij}}{\sum_{j'} \text{attn}_{ij'} + \epsilon}
\tag{4.9}
$$

## 4.8 Dropout

Standard dropout on attention weights:

$$
\text{attn}_{ij} \leftarrow \text{Dropout}(\text{attn}_{ij}, p)
\tag{4.10}
$$

This is the "dropout regularization" part of PADRe.

## 4.9 Context and Output

$$
\text{Context} = \text{attn} \cdot V
\tag{4.11}
$$

For nested encoders (RoBERTa), the context is returned directly. For flat 
encoders and decoders, it's passed through `out_proj`:

$$
\text{Output} = W_O \cdot \text{Context} + b_O
\tag{4.12}
$$

## 4.10 Full Algorithm

**Algorithm 1:** PADRe forward
```
Input:  X ∈ R^{B×N×d}, attention_mask, position_embeddings
        order d, temperature T, is_decoder
Output: Y ∈ R^{B×N×d}

 1: Q ← q_proj(X); K ← k_proj(X); V ← v_proj(X)
 2: if RoPE: (Q, K) ← apply_rotary(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: s ← QK^T / √d_h
 5: invalid ← pad_mask | causal_mask
 6: m ← max(s.masked_fill(invalid, -1e4), dim=-1)
 7: m ← where(m < -5e3, 0, m)                        ▷ All-masked guard
 8: s_centered ← (s - m) / T
 9: base ← ReLU(1 + s_centered / d)
10: attn ← base^d
11: attn ← attn.masked_fill(invalid, 0)
12: denom ← attn.sum(dim=-1, keepdim=True) + 1e-8
13: attn ← attn / denom
14: attn ← dropout(attn)
15: context ← attn @ V
16: Y ← out_proj(context)
17: return Y
```

## 4.11 Complexity Analysis

### 4.11.1 Time Complexity

| Operation | Complexity |
|---|---|
| Q/K/V projections | $\mathcal{O}(N d^2)$ |
| Scores | $\mathcal{O}(N^2 d_h)$ |
| Polynomial | $\mathcal{O}(N^2)$ |
| Normalize | $\mathcal{O}(N^2)$ |
| Context | $\mathcal{O}(N^2 d_h)$ |
| Output | $\mathcal{O}(N d^2)$ |
| **Total** | $\mathcal{O}(N^2 d_h)$ |

**This is identical to standard softmax attention.**

### 4.11.2 Kernelized Complexity (Future)

If we implement the kernelization:

$$
\text{attn} \cdot V = \frac{\phi(Q) \cdot (\phi(K)^\top V)}{\phi(Q) \cdot \phi(K)^\top \mathbf{1}}
\tag{4.13}
$$

The complexity becomes:

$$
\mathcal{O}(N \cdot F \cdot d_h)
\tag{4.14}
$$

where $F = \binom{d + d_h}{d}$ is the feature dimension.

**Not implemented in this version.**

## 4.12 Parameter Analysis

PADRe has the same parameter count as standard attention (plus LoRA adapters):

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (4 proj) | $4 r_L (d + d_h)$ |

For $d = 768$, $r_L = 16$: ~98K trainable per layer.

**No parameter reduction.**

## 4.13 Theoretical Notes

### 4.13.1 Taylor Approximation Error

For $|s| \ll d$:

$$
\left(1 + \frac{s}{d}\right)^d = e^s \cdot (1 + \mathcal{O}(s^2/d))
\tag{4.15}
$$

Error decreases as $d$ grows.

### 4.13.2 Centering and Stability

**Claim.** After max-centering, the maximum value of the base is 1.

**Proof.** $s_{ij} \leq 0$ after centering, so $1 + s_{ij}/d \leq 1$. $\square$

### 4.13.3 ReLU Sparsity

**Claim.** Positions with $s_{ij} < -d$ get zero attention weight.

**Proof.** ReLU$(1 + s_{ij}/d) = 0$ when $s_{ij} < -d$. $\square$

### 4.13.4 Causality

**Claim.** For decoders, causality is preserved.

**Proof.** The causal mask sets `invalid_mask[i, j] = True` for $j > i$. 
These positions are zeroed after the polynomial step, so they don't 
contribute to the context. $\square$

---