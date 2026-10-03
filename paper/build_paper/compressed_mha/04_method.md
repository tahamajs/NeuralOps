# 4. Method

## 4.1 Overview

Compressed MHA has four stages:

1. **Weight extraction** (§4.2): extract Q, K, V, O from pretrained attention.
2. **Joint QKV SVD** (§4.3): stack and factorize Q, K, V.
3. **Separate O SVD** (§4.4): factorize output projection.
4. **LoRA recovery** (§4.5): add trainable adapters on both branches.

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ Joint QKV SVD        │
    │  Vh_qkv: (r, d)      │  ← shared input basis
    │  S_qkv: (r,)         │
    │  U_qkv: (3d, r)      │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ QKV LoRA             │
    │  lora_A_qkv, lora_B_qkv │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Attention            │
    │  softmax(Q K^T) V    │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Separate O SVD       │
    │  Vh_out, S_out, U_out│
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ O LoRA               │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 Weight Extraction

We extract weights via `copy_weights`, handling four patterns:

### 4.2.1 DistilBERT (Flat)

```python
w_q = src.q_lin.weight
w_k = src.k_lin.weight
w_v = src.v_lin.weight
w_o = src.out_lin.weight
```

**Nested detection:** `out_lin` exists → `is_nested_encoder = False`.

### 4.2.2 RoBERTa (Nested)

```python
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested_encoder = True
    w_q = src.self.query.weight
    w_k = src.self.key.weight
    w_v = src.self.value.weight
    w_o = src.output.dense.weight
```

**Nested detection:** No `out_lin`/`o_proj`/`c_proj` → `is_nested_encoder = True`.

### 4.2.3 GPT-2 (Conv1D)

```python
w = src.c_attn.weight.t()  # Conv1D stored as (in, out)
q_w, k_w, v_w = w.chunk(3, dim=0)
w_o = src.c_proj.weight.t()
```

### 4.2.4 Qwen (GQA)

```python
w_q = src.q_proj.weight
w_k = src.k_proj.weight
w_v = src.v_proj.weight
w_o = src.o_proj.weight
```

## 4.3 Joint QKV SVD

### 4.3.1 Stacking

We concatenate along the output dimension:

$$
W_{QKV} = \begin{pmatrix} W_Q \\ W_K \\ W_V \end{pmatrix}
\tag{4.1}
$$

Shape: $[H d_h + 2 H_{kv} d_h, d]$.

### 4.3.2 SVD

Compute the full SVD:

$$
W_{QKV} = U_{QKV} \Sigma_{QKV} V_{QKV}^\top
\tag{4.2}
$$

Truncate to rank $r_{qkv}$:

$$
r_{qkv} = \min(\text{rank\_qkv}, H d_h + 2 H_{kv} d_h, d)
\tag{4.3}
$$

Store the factors:

$$
U_{QKV, r} \in \mathbb{R}^{(H d_h + 2 H_{kv} d_h) \times r_{qkv}}
\tag{4.4}
$$

$$
S_{QKV, r} \in \mathbb{R}^{r_{qkv}}, \quad
V_{QKV, r} \in \mathbb{R}^{r_{qkv} \times d}
\tag{4.5}
$$

### 4.3.3 Forward Pass

Given input $X$:

$$
\text{Proj}_1 = X V_{QKV, r}^\top \in \mathbb{R}^{B \times N \times r_{qkv}}
\tag{4.6}
$$

$$
\text{Proj}_2 = \text{Proj}_1 \cdot \text{diag}(S_{QKV, r}) \in \mathbb{R}^{B \times N \times r_{qkv}}
\tag{4.7}
$$

$$
\text{QKV} = \text{Proj}_2 U_{QKV, r}^\top \in \mathbb{R}^{B \times N \times (H d_h + 2 H_{kv} d_h)}
\tag{4.8}
$$

Then split into Q, K, V.

### 4.3.4 Parameter Count

| Component | Parameters |
|---|---|
| $V_{QKV, r}$ | $r_{qkv} \cdot d$ |
| $S_{QKV, r}$ | $r_{qkv}$ |
| $U_{QKV, r}$ | $r_{qkv} \cdot (H d_h + 2 H_{kv} d_h)$ |
| **Total** | $r_{qkv} (d + H d_h + 2 H_{kv} d_h) + r_{qkv}$ |

For $H = H_{kv}$:
$$
\text{Total}_{QKV} = r_{qkv} (2d + 2 d_h) = 2 r_{qkv} (d + d_h)
\tag{4.9}
$$

## 4.4 Separate O SVD

The output projection $W_O \in \mathbb{R}^{d \times H d_h}$ gets its own SVD:

$$
W_O = U_O \Sigma_O V_O^\top
\tag{4.10}
$$

Truncate to rank $r_{out}$:

$$
r_{out} = \min(\text{rank\_out}, d, H d_h)
\tag{4.11}
$$

Forward:

$$
\text{Out}_1 = \text{Context} V_{O, r}^\top \in \mathbb{R}^{B \times N \times r_{out}}
\tag{4.12}
$$

$$
\text{Out}_2 = \text{Out}_1 \cdot \text{diag}(S_{O, r}) \in \mathbb{R}^{B \times N \times r_{out}}
\tag{4.13}
$$

$$
\text{Out} = \text{Out}_2 U_{O, r}^\top + b_O \in \mathbb{R}^{B \times N \times d}
\tag{4.14}
$$

**Parameter count:**
$$
\text{Total}_O = r_{out} (d + H d_h) + r_{out}
\tag{4.15}
$$

## 4.5 LoRA Recovery

Both branches have LoRA adapters:

### 4.5.1 QKV LoRA

$$
\text{QKV}_\text{LoRA} = \frac{\alpha}{r_L} B_{QKV} A_{QKV} \cdot X
\tag{4.16}
$$

Added to the SVD output:

$$
\text{QKV}_\text{final} = \text{QKV}_\text{SVD} + \text{QKV}_\text{LoRA}
\tag{4.17}
$$

### 4.5.2 Output LoRA

$$
\text{Out}_\text{LoRA} = \frac{\alpha}{r_L} B_O A_O \cdot \text{Context}
\tag{4.18}
$$

Added after SVD output.

### 4.5.3 Trainable Parameters

| Component | Params |
|---|---|
| $A_{QKV}$ | $r_L \cdot d$ |
| $B_{QKV}$ | $(H d_h + 2 H_{kv} d_h) \cdot r_L$ |
| $A_O$ | $r_L \cdot H d_h$ |
| $B_O$ | $d \cdot r_L$ |
| **Total** | $r_L (2d + 2 H d_h)$ |

## 4.6 The Full Forward Pass

**Algorithm 1:** Compressed MHA forward
```
Input:  X ∈ R^{B×N×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×N×d}

 1: Proj_1 ← X @ Vh_qkv^T
 2: Proj_2 ← Proj_1 * S_qkv
 3: QKV ← Proj_2 @ U_qkv^T
 4: if lora_rank > 0:
 5:     QKV ← QKV + lora_B_qkv(lora_A_qkv(X)) * scaling
 6: Q, K, V ← split(QKV)
 7: Q ← Q + bias_q; K ← K + bias_k; V ← V + bias_v
 8: if RoPE: Q, K ← RoPE(Q, K)
 9: if GQA: K, V ← repeat_kv(K, V)
10: scores ← Q @ K^T / √d_h
11: if causal: mask(scores)
12: if padding: mask(scores)
13: Attn ← softmax(scores)
14: Context ← Attn @ V
15: if is_nested_encoder:
16:     return Context  # outer layer applies W_O
17: Out_1 ← Context @ Vh_out^T
18: Out_2 ← Out_1 * S_out
19: Out ← Out_2 @ U_out^T + bias_out
20: if lora_rank > 0:
21:     Out ← Out + lora_B_out(lora_A_out(Context)) * scaling
22: return Out
```

## 4.7 Nested Encoder Handling

**Critical detail.** For nested encoders (RoBERTa), our module **must not** 
apply its own output projection:

```python
if self.is_nested_encoder:
    out = context
else:
    # Apply SVD output projection
    out = ...
```

The outer `BertSelfOutput.dense` will apply the projection.

**Detection rule (in `copy_weights`):**

```python
# Check for flat output projections FIRST
if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
    self.is_nested_encoder = False
elif hasattr(src, "self") and hasattr(src, "output"):
    self.is_nested_encoder = True
else:
    self.is_nested_encoder = False
```

**Why order matters.** DistilBERT's `MultiHeadSelfAttention` ends with 
"SelfAttention" and has `out_lin`. An earlier implementation checked the 
class name **first**, mistakenly marking DistilBERT as nested → double 
projection → 50.92% collapse. This version checks `out_lin` first.

## 4.8 Complexity Analysis

### 4.8.1 Parameter Count

| Component | Baseline | Compressed | Ratio |
|---|---|---|---|
| $W_Q, W_K, W_V$ | $3 d^2$ (approx) | $2 r_{qkv} (d + d_h)$ | depends |
| $W_O$ | $d^2$ | $r_{out} (d + d_h)$ | depends |
| LoRA | 0 | $r_L (2d + 2 d_h)$ | +small |
| **Total** | $4 d^2$ | $2 r_{qkv} (d+d_h) + r_{out}(d+d_h) + r_L(2d + 2d_h)$ | — |

For $d = 768$, $d_h = 64$, $r_{qkv} = r_{out} = 384$, $r_L = 16$:

| Component | Params |
|---|---|
| Baseline | 2,359,296 |
| Compressed QKV | $2 \cdot 384 \cdot 832 = 638,976$ |
| Compressed O | $384 \cdot 832 = 319,488$ |
| LoRA | $16 \cdot 1664 = 26,624$ |
| **Total compressed** | ~985,088 |
| **Reduction** | **2.4×** |

For $r_{qkv} = r_{out} = 64$:
- Compressed QKV: $2 \cdot 64 \cdot 832 = 106,496$
- Compressed O: $64 \cdot 832 = 53,248$
- LoRA: 26,624
- **Total:** ~186,368 → **12.7× reduction**

### 4.8.2 Inference Complexity

| Operation | Complexity |
|---|---|
| Down-projection QKV | $\mathcal{O}(N r_{qkv} d)$ |
| Up-projection QKV | $\mathcal{O}(N r_{qkv} (d + d_h))$ |
| Attention | $\mathcal{O}(N^2 d_h)$ |
| Down-projection O | $\mathcal{O}(N r_{out} d_h)$ |
| Up-projection O | $\mathcal{O}(N r_{out} d)$ |

The attention dominates for long sequences.

## 4.9 Theoretical Notes

### 4.9.1 Joint SVD Optimality

**Claim.** The joint SVD of stacked $[W_Q; W_K; W_V]$ gives the minimum 
Frobenius error for any rank-$r$ approximation of the stacked matrix.

**Proof.** Direct application of Eckart-Young to $W_{QKV}$. $\square$

### 4.9.2 Shared Basis

**Claim.** The shared basis $V_{QKV, r}$ captures the common structure of Q, 
K, V in the input space.

**Proof sketch.** If $W_Q \approx W_Q V V^\top$ for a shared subspace $V$, the 
joint SVD recovers $V$ as the dominant right singular vectors. $\square$

### 4.9.3 Reconstruction Error

For the joint SVD:

$$
\|W_{QKV} - U_{QKV, r} \Sigma_r V_{QKV, r}^\top\|_F = \sqrt{\sum_{i > r} \sigma_i^2}
\tag{4.19}
$$

### 4.9.4 Causality

**Claim.** The causal mask is preserved.

**Proof.** We apply a triangular mask before softmax, standard for causal 
attention. $\square$

---