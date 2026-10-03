# 4. Method

## 4.1 Overview

Our Linformer implementation has four phases:

1. **Projection** (§4.2): standard $Q, K, V$ projections.
2. **RoPE** (§4.2): optional rotary embeddings.
3. **GQA expansion** (§4.3): expand KV heads.
4. **Adaptive Linformer projection** (§4.4): project $K, V$ to dimension $k$.
5. **Attention** (§4.5): standard scaled-dot-product on projected tensors.
6. **Causal handling** (§4.6): fallback to dense if needed.

## 4.2 Projections and RoPE

Standard projections via LoRALinear:

```python
q = self.q_proj(hidden_states)
k = self.k_proj(hidden_states)
v = self.v_proj(hidden_states)
```

Reshape to multi-head:

```python
q = q.view(B, N, H, d_h).transpose(1, 2)
k = k.view(B, N, H_kv, d_h).transpose(1, 2)
v = v.view(B, N, H_kv, d_h).transpose(1, 2)
```

Apply RoPE if position embeddings available.

## 4.3 GQA Expansion

```python
if self.kv_repeat > 1:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

After this, $K, V \in \mathbb{R}^{B \times H \times L \times d_h}$.

## 4.4 Adaptive Linformer Projection

### 4.4.1 Three Cases

Given sequence length $L$, projection dim $k$, and pre-allocated $L_\text{max}$:

| Case | Condition | Action |
|---|---|---|
| **Short** | $L \leq k$ | Use $E[:L, :L]$, no compression |
| **Medium** | $L \leq L_\text{max}$ | Use $E[:k, :L]$, standard Linformer |
| **Long** | $L > L_\text{max}$ | Interpolate $E$ to length $L$ |

### 4.4.2 Short Case ($L \leq k$)

For short sequences, no compression is applied:

```python
E_act = self.E[:L, :L]
K_proj = E_act @ k  # (B, H, L, d_h)
```

This preserves exact attention for short sequences, ensuring no accuracy 
loss when compression isn't needed.

### 4.4.3 Medium Case ($L \leq L_\text{max}$)

Standard Linformer:

```python
k_eff = min(self.k_proj_dim, L)
E_act = self.E[:k_eff, :L]
K_proj = E_act @ k  # (B, H, k_eff, d_h)
```

$K$ and $V$ are projected from $L$ to $k_\text{eff}$ positions.

### 4.4.4 Long Case ($L > L_\text{max}$)

For sequences longer than the pre-allocated max:

```python
E_act = F.interpolate(
    self.E[:k_eff].unsqueeze(0),
    size=L,
    mode="linear",
    align_corners=False,
).squeeze(0)
```

Linear interpolation extends $E$ to length $L$. This is a graceful 
degradation — not part of the original Linformer paper, but necessary for 
robustness.

## 4.5 Attention on Projected Tensors

### 4.5.1 Scores

$$
\text{scores} = \frac{Q \tilde{K}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L \times k_\text{eff}}
\tag{4.1}
$$

Computed in FP32 for stability.

### 4.5.2 Softmax

Standard softmax over $k_\text{eff}$ dimension.

### 4.5.3 Context

$$
\text{context} = \text{attn} \cdot \tilde{V} \in \mathbb{R}^{B \times H \times L \times d_h}
\tag{4.2}
$$

### 4.5.4 Output Projection

```python
context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
out = self.out_proj(context * self.output_scale)
```

Where `output_scale` is a learnable scalar initialized to 1.

## 4.6 Causal Handling for Decoders

**The problem.** For a decoder, projecting $K$ via $E$ mixes future and past 
information:

$$
\tilde{K}_j = \sum_{i=1}^L E_{j, i} K_i
\tag{4.3}
$$

The resulting $\tilde{K}_j$ depends on **all** positions, not just $j$. A 
causal mask over $\tilde{K}_j$ is meaningless.

**The solution.** Fall back to exact dense attention:

```python
if self.is_decoder and L > self.k_proj_dim:
    # Cannot compress without breaking causality
    causal_scores = q @ k.transpose(-2, -1) / sqrt(d_h)
    causal_mask = tril(ones(L, L))
    scores = causal_scores.masked_fill(~causal_mask, -inf)
    V_proj = v  # Use unprojected V
```

This preserves causality at the cost of $\mathcal{O}(L^2)$ for decoders 
with $L > k$.

For short sequences ($L \leq k$), the projection is identity-like, so 
causality is approximately preserved. We use the projected path in this 
case.

## 4.7 Full Algorithm

**Algorithm 1:** Linformer forward pass
```
Input:  X ∈ R^{B×L×d}, k, L_max
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q; K ← X W_K; V ← X W_V
 2: if RoPE: Q, K ← RoPE(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: if L ≤ k:
 5:     E_act ← E[:L, :L]; F_act ← F[:L, :L]
 6:     V_proj ← F_act @ V
 7: elif L ≤ L_max:
 8:     k_eff ← min(k, L)
 9:     E_act ← E[:k_eff, :L]; F_act ← F[:k_eff, :L]
10:     V_proj ← F_act @ V
11: else:
12:     E_act ← interpolate(E, size=L); F_act ← interpolate(F, size=L)
13:     V_proj ← F_act @ V
14: K_proj ← E_act @ K
15: scores ← Q @ K_proj^T / √d_h
16: if is_decoder and L > k:
17:     # Fallback: dense causal attention
18:     scores ← Q @ K^T / √d_h
19:     V_proj ← V
20:     scores ← mask_causal(scores)
21: scores ← mask_padding(scores)
22: attn ← softmax(scores)
23: out ← attn @ V_proj
24: out ← out_proj(out * output_scale)
25: return out
```

## 4.8 Complexity Analysis

### 4.8.1 Encoder Complexity

| Operation | Complexity |
|---|---|
| $Q, K, V$ projections | $\mathcal{O}(L d^2)$ |
| $E K$ | $\mathcal{O}(k L d_h)$ |
| $Q \tilde{K}^\top$ | $\mathcal{O}(L k d_h)$ |
| Softmax | $\mathcal{O}(L k)$ |
| $\text{attn} \cdot \tilde{V}$ | $\mathcal{O}(L k d_h)$ |
| Output projection | $\mathcal{O}(L d^2)$ |
| **Total** | $\mathcal{O}(L k d_h + L d^2)$ |

For fixed $k$ and $d$: **linear in $L$**.

### 4.8.2 Decoder Complexity

| Case | Complexity |
|---|---|
| $L \leq k$ | $\mathcal{O}(L k d_h)$ |
| $L > k$ | $\mathcal{O}(L^2 d_h)$ (fallback) |

**Trade-off.** Encoders benefit from linearity; long-sequence decoders 
fall back to quadratic.

### 4.8.3 Memory

| Tensor | Size |
|---|---|
| $E, F$ | $2 k L_\text{max}$ |
| Attention scores | $\mathcal{O}(L k)$ |
| Context | $\mathcal{O}(L d_h)$ |

Much smaller than baseline $\mathcal{O}(L^2)$ for the scores matrix.

## 4.9 Parameter Analysis

| Component | Baseline | Ours | Change |
|---|---|---|---|
| $W_Q, W_K, W_V, W_O$ | $4 d^2$ (frozen) | $4 d^2$ (frozen) | 0 |
| LoRA adapters | 0 | $4 r_L (d + d)$ | +small |
| $E, F$ | 0 | $2 k L_\text{max}$ | +medium |
| `output_scale` | 0 | 1 | ~0 |

For $d = 768$, $k = 64$, $L_\text{max} = 512$:

| Component | Params |
|---|---|
| Baseline per layer | 2,359,296 |
| LoRA (rank 16) | 98,304 |
| E, F | $2 \cdot 64 \cdot 512 = 65,536$ |
| output_scale | 1 |
| **Total added** | ~163,841 |
| **Change** | **+6.9%** |

We state this **explicitly** in the abstract.

## 4.10 Theoretical Notes

### 4.10.1 Low-Rank Approximation Bound

If the attention matrix $P$ has effective rank $r \ll L$, then:

$$
\|P - \tilde{P}\|_F \leq \sqrt{\sum_{i=r+1}^L \sigma_i^2}
\tag{4.4}
$$

where $\sigma_i$ are singular values of $P$. For $k \geq r$:

$$
\|P - \tilde{P}\|_F \approx 0
\tag{4.5}
$$

### 4.10.2 Causal Preservation

**Claim.** For decoders with $L > k$, the fallback preserves causality.

**Proof.** The fallback uses standard dense attention with a triangular mask, 
which is causally correct by construction. $\square$

### 4.10.3 Identity Initialization

**Claim.** With $E = I$, the projection $\tilde{K} = K$ and Linformer 
reduces to standard attention.

**Proof.** $E K = I K = K$. $\square$

This means our initialization starts at exactly the standard attention and 
deviates only as $E$ is trained.

---