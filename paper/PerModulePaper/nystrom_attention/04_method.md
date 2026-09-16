<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The algorithm
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

This section describes our reproduction of Nyströmformer from Xiong et al. 
(2021). We do not introduce novel components.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input hidden_states ∈ R^{B × N × D}         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Q = query(x), K = key(x), V = value(x)      │
   │  (with LoRA adapters)                         │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  RoPE (if applicable)                         │
   │  GQA repeat_kv (if applicable)                │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  Branching:                                    │
   │  - if decoder: dense causal attention         │
   │  - if N ≤ 128: dense attention                │
   │  - else: Nyström branch                        │
   └────────────────────┬─────────────────────────┘
                        │
                        ▼
   ┌──────────────────────────────────────────────┐
   │  out_proj (if not nested)                     │
   └──────────────────────────────────────────────┘
```

## 4.2 Three Execution Paths

Nyströmformer in our implementation has **three paths**:

### 4.2.1 Causal Decoder Path

For autoregressive decoders (GPT-2, Qwen3.5), Nyström approximation is 
not directly applicable because landmark pooling averages across tokens 
including future ones. We fall back to **dense causal attention**.

### 4.2.2 Short Sequence Fallback

For $N \leq \max(128, 2m)$, we use dense attention. Reasons:

- Nyström approximation is sub-optimal when $N$ is small.
- Dense attention is memory-efficient for short sequences.

### 4.2.3 Nyström Path

For $N > \max(128, 2m)$ and non-causal attention:

$$
\text{Output} = \text{softmax}(Q K_m^\top) \cdot \text{pinv}(A_{mm}) \cdot \text{softmax}(Q_m K^\top) V
\tag{4.1}
$$

where $A_{mm} = \text{softmax}(Q_m K_m^\top / \sqrt{d_h})$.

## 4.3 Landmark Selection

We use `adaptive_avg_pool1d`:

```python
Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)   # (BH, d, N)
Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
```

Same for $K$.

**Properties:**
- Deterministic.
- $\mathcal{O}(N d_h)$ cost.
- Adjacent tokens share landmarks.

## 4.4 The Three Softmax Matrices

$$
\begin{aligned}
F_1 &= \text{softmax}(Q K_m^\top / \sqrt{d_h}) \in \mathbb{R}^{N \times m} \\
F_2 &= \text{softmax}(Q_m K_m^\top / \sqrt{d_h}) \in \mathbb{R}^{m \times m} \\
F_3 &= \text{softmax}(Q_m K^\top / \sqrt{d_h}) \in \mathbb{R}^{m \times N}
\end{aligned}
\tag{4.2}
$$

Then:

$$
\text{Output} = F_1 \cdot \text{pinv}(F_2) \cdot (F_3 V)
\tag{4.3}
$$

**Associative order:** Compute $F_3 V$ first ($m \times d_h$), then pinv 
($m \times m$), then $F_1$ ($N \times m$). This ensures $\mathcal{O}(N m d_h)$ 
total.

## 4.5 Iterative Pseudoinverse

Nyströmformer uses Newton-Schulz iteration:

```python
def _iterative_pinv(A, iters=3):
    try:
        return torch.linalg.pinv(A.float()).to(A.dtype)
    except Exception:
        # Newton-Schulz fallback
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V
```

**Note.** Our implementation first tries `torch.linalg.pinv` (SVD-based). 
If it fails (e.g., singularity), we fall back to Newton-Schulz.

**Difference from original.** The original Nyströmformer uses Newton-Schulz 
exclusively. We use direct pinv when possible, which is more accurate.

## 4.6 Weight Transfer from Pretrained Attention

### 4.6.1 Architecture-Aware Extraction

| Architecture | Extraction |
|---|---|
| DistilBERT | `q_lin, k_lin, v_lin, out_lin` |
| RoBERTa | `self.query, self.key, self.value, output.dense` |
| GPT-2 | Split `c_attn`; `c_proj` for output |
| Qwen3.5 | `q_proj, k_proj, v_proj, o_proj` |

### 4.6.2 Conv1D Transposition

```python
if is_conv1d:
    c_w = c_w.t()
```

### 4.6.3 Nested Encoder Bypass

```python
if self.is_nested_encoder:
    out = context   # Skip out_proj
else:
    out = self.out_proj(context)
```

### 4.6.4 GQA

```python
if self.kv_repeat > 1:
    K = repeat_kv(K, self.kv_repeat)
    V = repeat_kv(V, self.kv_repeat)
```

## 4.7 The Full Algorithm

**Algorithm 1:** Nyström attention forward
```
Input:  Q, K, V ∈ R^{B × H × N × d}
        valid_mask ∈ {0,1}^{B × N}
        num_landmarks m
Output: context ∈ R^{B × H × N × d}

1:  if is_causal or is_decoder:
2:      # Dense causal attention
3:      A ← softmax(Q K^T / √d + causal_mask + valid_mask)
4:      context ← A V
5:      return context
6:
7:  if N ≤ max(128, 2m):
8:      # Dense attention
9:      A ← softmax(Q K^T / √d + valid_mask)
10:     context ← A V
11:     return context
12:
13: # Nyström branch
14: Q_m ← adaptive_avg_pool(Q, m)
15: K_m ← adaptive_avg_pool(K, m)
16:
17: F1 ← softmax(Q K_m^T / √d)         # (B, H, N, m)
18: F2 ← softmax(Q_m K_m^T / √d)       # (B, H, m, m)
19: F3 ← softmax(Q_m K^T / √d)         # (B, H, m, N)
20:
21: F2_pinv ← iterative_pinv(F2)       # (B, H, m, m)
22: step1 ← F3 V                        # (B, H, m, d)
23: step2 ← F2_pinv step1               # (B, H, m, d)
24: context ← F1 step2                  # (B, H, N, d)
25:
26: return context
```

## 4.8 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Dense causal | $\mathcal{O}(N^2 d_h)$ | $\mathcal{O}(N^2)$ |
| Dense (short) | $\mathcal{O}(N^2 d_h)$ | $\mathcal{O}(N^2)$ |
| Nyström | $\mathcal{O}(N m d_h)$ | $\mathcal{O}(N m)$ |

**Nyström speedup:**

$$
\frac{\text{Dense}}{\text{Nyström}} = \frac{N^2 d_h}{N m d_h} = \frac{N}{m}
\tag{4.4}
$$

For $N = 4096$, $m = 32$: **128× speedup**.

## 4.9 Theoretical Notes

### 4.9.1 Approximation Error

**Claim.** If the attention matrix $A$ has rank $\leq m$, the Nyström 
approximation is exact.

**Proof.** Nyström reconstruction is exact for rank-$m$ matrices. $\square$

In practice, $A$ is approximately low-rank. Error depends on the tail of 
the singular value spectrum.

### 4.9.2 Newton-Schulz Convergence

**Proposition.** The iteration $V_{k+1} = V_k (2I - A V_k)$ with 
$V_0 = \alpha A^\top$ converges to $A^+$ if $\|I - \alpha A A^\top\|_2 < 1$.

**Proof.** Classical. $\square$

### 4.9.3 Adaptive Pooling as Landmark Selection

**Observation.** Adaptive pooling selects landmarks as segment averages. 
This is equivalent to sampling at stride $N / m$ with local averaging.

**Comparison to k-means:** Adaptive pooling is deterministic and faster. 
K-means would give better landmarks at higher cost.

## 4.10 Summary

Nyströmformer:
1. Approximates attention via landmarks.
2. Uses adaptive pooling for landmarks.
3. Uses Newton-Schulz pinv (or direct pinv).
4. Falls back to dense for short sequences and decoders.
5. Does **not** reduce parameter count or KV cache.
6. Reduces attention **compute**.

---