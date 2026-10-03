<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: Method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

PTP is a wrapper for standard attention that replaces the full $L \times L$ 
attention with a pruned $L \times k$ attention:

1. **Score** each token with a small MLP.
2. **Select** the top-$k$ tokens.
3. **Prune** K, V to the selected tokens.
4. **Attend** using the pruned K, V.

**Figure 4.1** illustrates the pipeline.

```
    X ∈ R^{B × L × d}
          │
          ├──────────────────┐
          ▼                  ▼
   ┌──────────────┐    ┌─────────────┐
   │ Score MLP    │    │ Q/K/V proj  │
   │ s = σ(MLP(X))│    └──────┬──────┘
   └──────┬───────┘           │
          ▼                   │
   ┌──────────────┐           │
   │ Top-k        │           │
   │ T = top_k(s) │           │
   └──────┬───────┘           │
          │                   │
          ▼                   ▼
   ┌──────────────┐    ┌─────────────┐
   │ Prune K, V   │◄───│ Gather      │
   └──────┬───────┘    └─────────────┘
          │
          ▼
   ┌──────────────┐
   │ Attention    │
   │ Q · K_T^T    │
   └──────────────┘
```

## 4.2 Score Network

### 4.2.1 Architecture

$$
s_i = \sigma\!\left(W_2 \cdot \text{GELU}(W_1 x_i + b_1) + b_2\right)
\tag{4.1}
$$

with:
- $W_1 \in \mathbb{R}^{64 \times d}$: input projection.
- $W_2 \in \mathbb{R}^{1 \times 64}$: output scalar.
- $\sigma$: sigmoid.

**Parameters:** $64 (d + 1) + 1 \approx 64 d$.

For $d = 768$: ~49K parameters per attention layer.

### 4.2.2 Initialization

- $W_1$: Kaiming uniform.
- $W_2$: Kaiming uniform.
- Bias: zero.

At initialization, scores are roughly uniform in $(0, 1)$.

### 4.2.3 Why a Small MLP?

Alternatives:

| Type | Pros | Cons |
|---|---|---|
| Linear ($w^\top x$) | Minimal params | Cannot model nonlinearity |
| **MLP-64** | **Expressive, small** | **~50K params/layer** |
| MLP-256 | More expressive | ~200K params/layer |

MLP-64 is a good trade-off.

## 4.3 Token Selection

### 4.3.1 [CLS] Preservation

Set $s_0 \leftarrow 10^4$ to guarantee retention:

```python
scores[:, 0] = 1e4
```

### 4.3.2 Padding Masking

Set padding scores to $-10^4$:

```python
valid_mask = get_token_valid_mask(attention_mask, B, L, device)
scores = scores.masked_fill(~valid_mask, -1e4)
```

### 4.3.3 Order of Operations

**Critical:** apply padding masking **after** [CLS] preservation so that 
if [CLS] is padding (it never should be), it is still excluded.

```python
scores[:, 0] = 1e4               # CLS preservation
scores = scores.masked_fill(~valid_mask, -1e4)  # Padding
scores[:, 0] = 1e4               # Re-apply CLS (padding mask may have overwritten)
```

### 4.3.4 Top-K Selection

$$
k = \lceil \rho L \rceil, \quad
\mathcal{T} = \text{top-}k(s)
\tag{4.2}
$$

We sort the indices for consistent gathering:

```python
k_tokens = max(1, min(L, int(math.ceil(L * self.keep_ratio))))
_, top_indices = torch.topk(scores, k_tokens, dim=-1)
top_indices, _ = torch.sort(top_indices, dim=-1)
```

Sorting ensures the pruned sequence preserves original order, which helps 
downstream layers.

## 4.4 KV Pruning

### 4.4.1 Gather Operation

Given $K_\text{all}, V_\text{all} \in \mathbb{R}^{B \times H \times L \times d_h}$ 
and indices $\mathcal{T} \in \mathbb{Z}^{B \times k}$:

$$
K_\text{pruned}[b, h, i, :] = K_\text{all}[b, h, \mathcal{T}[b, i], :]
\tag{4.3}
$$

Implemented via `torch.gather`:

```python
top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d_h)
k_pruned = torch.gather(k_all, 2, top_idx_exp)
v_pruned = torch.gather(v_all, 2, top_idx_exp)
```

### 4.4.2 Complexity

| Operation | Complexity |
|---|---|
| Gather | $\mathcal{O}(B H k d_h)$ |
| Attention | $\mathcal{O}(B H L k d_h)$ |

**Reduction from dense:** $k / L = \rho$.

## 4.5 Attention on Pruned KV

$$
S = \frac{Q K_\text{pruned}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times L \times k}
\tag{4.4}
$$

### 4.5.1 Causal Masking

For decoders, queries can only attend to keys at positions $\leq$ query 
position. After pruning, we compare **original** positions:

$$
\text{mask}_{ij} = \mathbb{1}[\mathcal{T}[b, j] > i]
\tag{4.5}
$$

**Implementation:**

```python
q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
k_orig_pos = top_indices.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k)
causal_mask = k_orig_pos > q_pos
attn_scores = attn_scores.masked_fill(causal_mask, -1e4)
```

### 4.5.2 Padding Masking

We gather padding validity for the retained keys:

```python
valid_keys = torch.gather(valid_mask, 1, top_indices)  # (B, k)
attn_scores = attn_scores.masked_fill(~valid_keys.unsqueeze(1).unsqueeze(2), -1e4)
```

### 4.5.3 Softmax and Dropout

Standard:

$$
P = \text{softmax}(S), \quad P \leftarrow \text{Dropout}(P)
\tag{4.6}
$$

## 4.6 Output Projection

For non-nested encoders:

$$
O = P V_\text{pruned}, \quad \text{out} = W_O O + b_O
\tag{4.7}
$$

For nested encoders (RoBERTa/BERT), we skip $W_O$:

$$
\text{out} = O
\tag{4.8}
$$

## 4.7 Complete Forward Pass

**Algorithm 1:** PTP forward
```
Input:  hidden_states ∈ R^{B × L × d}
        attention_mask, position_embeddings, is_causal
        keep_ratio ρ
Output: output ∈ R^{B × L × d}

 1: # 1. Score tokens
 2: s ← sigmoid(MLP(hidden_states))          # (B, L)
 3: s[:, 0] ← 1e4                            # CLS preservation
 4: s ← s.masked_fill(~valid_mask, -1e4)     # Padding
 5: s[:, 0] ← 1e4                            # Re-apply CLS
 6:
 7: # 2. Top-k selection
 8: k ← ceil(ρ · L)
 9: T ← topk(s, k)
10: T ← sort(T)                             # Preserve order
11:
12: # 3. Project Q, K, V
13: Q ← W_Q(hidden_states)
14: K_all ← W_K(hidden_states)
15: V_all ← W_V(hidden_states)
16:
17: # 4. RoPE
18: if position_embeddings: Q, K_all ← RoPE(Q, K_all)
19:
20: # 5. GQA repeat
21: if H_kv < H: K_all, V_all ← repeat_kv(K_all, V_all, H/H_kv)
22:
23: # 6. Gather K, V at T
24: K ← gather(K_all, dim=2, T)
25: V ← gather(V_all, dim=2, T)
26:
27: # 7. Attention
28: S ← Q K^T / sqrt(d_h)
29: if is_causal: S ← S + causal_mask_from_T
30: S ← S + padding_mask_from_T
31: P ← softmax(S)
32: O ← P V
33:
34: # 8. Output projection
35: if nested_encoder: out ← O
36: else: out ← W_O(O)
37:
38: return out
```

## 4.8 Complexity Analysis

| Component | Dense | PTP |
|---|---|---|
| Score network | — | $\mathcal{O}(B L \cdot 64 d)$ |
| Top-$k$ | — | $\mathcal{O}(B L \log k)$ |
| Q/K/V projections | $\mathcal{O}(B L d^2)$ | Same |
| Attention | $\mathcal{O}(B H L^2 d_h)$ | **$\mathcal{O}(B H L k d_h)$** |
| Output projection | $\mathcal{O}(B L d^2)$ | Same |
| **Total attention** | $\mathcal{O}(B H L^2 d_h)$ | **$\mathcal{O}(B H L k d_h)$** |

**Reduction factor:** $L / k = 1 / \rho$.

## 4.9 Parameter Analysis

| Component | Baseline | PTP |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | $4 d^2$ | Same |
| Score network | 0 | $64 (d + 1) + 1 \approx 64 d$ |
| LoRA adapters | 0 | $2 r_L \cdot 4 d$ |

**Trainable:** Score network + LoRA = $64 d + 8 r_L d$ per layer.

For $d = 768, r_L = 16$: ~49K + 98K = **147K** trainable per layer.

## 4.10 Theoretical Notes

### 4.10.1 [CLS] Retention Guarantee

**Claim.** Token 0 is always in $\mathcal{T}$.

**Proof.** We set $s_0 = 10^4$ before top-$k$. Since all other scores are 
bounded in $(-10^4, 10^4)$, position 0 has the maximum score and is always 
selected. $\square$

### 4.10.2 Causality Guarantee

**Claim.** For decoder models, PTP never attends to future positions.

**Proof.** We mask $(i, j)$ whenever $\mathcal{T}[b, j] > i$. This ensures 
only original positions $\leq i$ are attended. $\square$

### 4.10.3 Approximation Error

**Claim.** The output of PTP attention is bounded by:

$$
\|O - O_\text{dense}\| \leq 2 \max_i \|V_i\| \cdot \sum_{j \notin \mathcal{T}} P_j
\tag{4.9}
$$

**Sketch.** The difference is the sum over pruned keys weighted by their 
attention probability. $\square$

### 4.10.4 Optimal Keep Ratio

**Claim.** Under the assumption that task accuracy is proportional to 
retained attention mass, the optimal $\rho$ satisfies:

$$
\frac{\partial}{\partial \rho} \left[\text{Acc}(\rho) - \lambda \cdot \rho\right] = 0
\tag{4.10}
$$

**Empirical observation.** $\rho^* \approx 0.5$ for SST-2.

---