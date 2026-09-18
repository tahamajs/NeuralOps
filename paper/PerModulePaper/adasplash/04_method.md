<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 – 5 pages (1500 – 2500 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

AdaSplash replaces the attention mechanism of a pretrained transformer with 
four modifications:

1. **α-Entmax normalization** (§4.2) replaces softmax.
2. **Local splash masking** (§4.3) restricts attention to a sliding window 
   plus sink tokens.
3. **Frozen base weights + LoRA** (§4.4) preserves pretrained representations 
   while providing trainable capacity.
4. **Multi-architecture weight transfer** (§4.5) enables drop-in replacement.

**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────────┐
    │  Pretrained attention A with Q, K, V, O  │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  AdaSplash attention A':                 │
    │    1. Project with frozen base + LoRA    │
    │    2. Apply splash mask                  │
    │    3. Apply α-entmax                     │
    │    4. Compute context, apply out_proj    │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │  Compressed attention with:              │
    │    - Frozen base weights (exact copy)    │
    │    - Trainable LoRA adapters (rank r_L)  │
    │    - α-entmax sparse activations         │
    │    - Local splash window                 │
    └──────────────────────────────────────────┘
```

## 4.2 Phase 1: α-Entmax Normalization

### 4.2.1 Problem Statement

Given scores $s \in \mathbb{R}^{L}$, compute:

$$
p_i = \max(0, (\alpha - 1) s_i - \tau)^{1/(\alpha - 1)}
\tag{4.1}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

with $\tau$ chosen so that $\sum_i p_i = 1$.

### 4.2.2 Bisection Algorithm

**Algorithm 1:** α-Entmax via bisection
```
Input: scores s ∈ R^{B×H×L_q×L_k}, α, valid_mask, n_iter
Output: p ∈ R^{B×H×L_q×L_k} with sum_i p_i = 1

1:  if |α - 1| < ε:                        ▷ Softmax case
2:      return softmax(s, masked)
3:
4:  s_masked ← mask_invalid(s, -10000)
5:  s_shift ← s_masked - max(s_masked)
6:  scaled ← s_shift * (α - 1)
7:
8:  τ_lo ← min(scaled) - 1
9:  τ_hi ← 0
10:
11: for _ in range(n_iter):
12:     τ ← (τ_lo + τ_hi) / 2
13:     p ← max(0, scaled - τ)^{1/(α-1)}
14:     p ← mask_invalid(p, 0)
15:     sum_p ← sum(p)
16:     if sum_p > 1:
17:         τ_lo ← τ
18:     else:
19:         τ_hi ← τ
20:
21: τ ← (τ_lo + τ_hi) / 2
22: p ← max(0, scaled - τ)^{1/(α-1)}
23: p ← mask_invalid(p, 0)
24: p ← p / sum(p)                          ▷ Normalize
25:
26: if sum(p) < ε:                           ▷ Support empty
27:     p ← softmax(s_masked)                 ▷ Fallback
28:
29: return p
```

**Convergence.** Bisection on $[\tau_\text{lo}, \tau_\text{hi}]$ converges 
to precision $(\tau_\text{hi} - \tau_\text{lo}) / 2^{n_\text{iter}}$. With 
25 iterations and initial range $O(1)$, precision is $O(10^{-7})$.

**Numerical stability.**

1. **Shift scores.** Subtract $\max_i s_i$ to bound $(\alpha - 1) s_i$.
2. **Mask invalid keys.** Set to $-10000$ before the algorithm.
3. **Softmax fallback.** If the support is empty (all p_i = 0), fall back 
   to softmax.

### 4.2.3 Complexity

- **Time:** $O(n_\text{iter} \cdot L^2)$ for sequence length $L$.
- **Memory:** $O(L^2)$ for the attention matrix.

For $L = 128$ and $n_\text{iter} = 25$: ~$4 \times 10^5$ operations per head, 
comparable to 25 softmaxes.

### 4.2.4 Sparsity Pattern

For $\alpha = 1.5$, the support size depends on the score distribution:

- **Uniform scores:** ~50% sparsity (barely sparse).
- **Peaked scores:** ~80% sparsity.
- **Real attention scores:** ~40% sparsity empirically.

**Figure 4.2** shows sparsity vs $\alpha$ for a typical attention layer.

## 4.3 Phase 2: Local Splash Masking

### 4.3.1 Splash Mask Definition

For causal decoders, we restrict attention to a local window of radius $W$ 
plus the first $S$ sink tokens:

$$
\text{splash}(i, j) = \begin{cases}
1 & \text{if } 0 \leq i - j \leq W \\
1 & \text{if } j < S \\
0 & \text{otherwise}
\end{cases}
\tag{4.2}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

Keys outside the splash mask are set to $-10000$ before entmax.

### 4.3.2 Non-Causal Models

For bidirectional encoders (DistilBERT, RoBERTa), we **do not** apply 
splash masking. The full attention is used with α-entmax. This preserves 
global context.

**Rationale.** Splash masking is a **decoder-specific** technique for 
long-context inference. On short sequences (L ≤ 128), the window radius 
$W = 32$ would restrict attention unnecessarily for encoders.

### 4.3.3 Interaction with Attention Sinks

The sink tokens ($j < S$, default $S = 4$) are always preserved, even outside 
the splash window. This maintains the stability property discovered by 
Xiao et al. (2024).

## 4.4 Phase 3: Frozen Base Weights + LoRA

### 4.4.1 Base Weight Transfer

For each projection $P \in \{Q, K, V, O\}$, we copy $W_P$ from the pretrained 
attention module. The exact copy procedure depends on the architecture:

- **DistilBERT:** Direct copy from `q_lin`, `k_lin`, `v_lin`, `out_lin`.
- **RoBERTa:** Copy from `self.query`, `self.key`, `self.value`; leave 
  `out_proj` identity (outer `output.dense` applies it).
- **GPT-2:** Split fused `c_attn.weight` (transposed) into Q, K, V; copy 
  `c_proj.weight`.
- **Qwen:** Direct copy from `q_proj`, `k_proj`, `v_proj`, `o_proj`.

### 4.4.2 Freezing Base

After copying, base weights are frozen (`requires_grad = False`). Biases are 
also frozen if present.

### 4.4.3 LoRA Adapters

For each projection, we add a rank-$r_L$ adapter:

$$
\Delta W_P = B_P A_P, \quad A_P \in \mathbb{R}^{r_L \times d_\text{in}}, \quad B_P \in \mathbb{R}^{d_\text{out} \times r_L}
\tag{4.3}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

with initialization:

$$
A_P \sim \mathcal{N}(0, \sigma^2), \quad B_P = 0
\tag{4.4}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

At initialization, $\Delta W_P = 0$, so the forward pass matches the 
pretrained model exactly.

### 4.4.4 Forward Pass

$$
\text{Proj}_P(x) = x W_P^\top + x (B_P A_P)^\top + b_P
\tag{4.5}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

$$
\text{Proj}_P(x) = x W_P^\top + (x A_P^\top) B_P^\top + b_P
\tag{4.6}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

### 4.4.5 Parameter Count

| Configuration | Parameters |
|---|---|
| Original | $d_\text{in} d_\text{out}$ |
| AdaSplash base (frozen) | $d_\text{in} d_\text{out}$ (same) |
| AdaSplash + LoRA | $d_\text{in} d_\text{out} + r_L (d_\text{in} + d_\text{out})$ |
| **Parameter change** | **+$r_L (d_\text{in} + d_\text{out})$** |

For $d = 768$, $r_L = 16$: each projection adds $16 \times 1536 = 24{,}576$ 
parameters. Across 4 projections × $N_l$ layers: significant addition.

**AdaSplash is not a compression method.**

## 4.5 Phase 4: Multi-Architecture Weight Transfer

### 4.5.1 Architecture Detection

```python
# Nested encoder detection
is_nested = (
    hasattr(src, "self") and hasattr(src, "output") and
    hasattr(src.self, "query")
)
```

For nested encoders, we extract $Q, K, V$ from `src.self` and set 
`out_proj` to identity (the outer `output.dense` applies the projection).

### 4.5.2 Frobenius Validation

After copying, we verify:

$$
\|W_P\|_F > 10^{-4} \quad \forall P \in \{Q, K, V\}
\tag{4.7}

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->

<!-- source: Codes/framework/attentions/adasplash_attention.py line 195 -->
$$

If a projection has near-zero norm, we raise an error and skip injection.

## 4.6 The Full Algorithm

**Algorithm 2:** AdaSplash forward pass
```
Input:  hidden_states X ∈ R^{B×L×d}
        attention_mask, position_embeddings
        hyperparameters α, W, S, r_L
Output: output Y ∈ R^{B×L×d}

 1: Q ← X W_Q^T + (X A_Q^T) B_Q^T
 2: K ← X W_K^T + (X A_K^T) B_K^T
 3: V ← X W_V^T + (X A_V^T) B_V^T
 4:
 5: if position_embeddings: (Q, K) ← apply_rotary(Q, K, cos, sin)
 6: if H_kv < H: K, V ← repeat_kv(K, V)
 7:
 8: S ← (Q @ K^T) / √d_h
 9: S ← apply_attention_mask(S, mask)
10:
11: if is_decoder:
12:     causal_mask ← tril(ones(L, L))
13:     splash_mask ← (0 ≤ i - j ≤ W) | (j < S)
14:     S ← S.masked_fill(~(causal & splash), -10000)
15:
16: A ← α_entmax(S, α)
17: A ← dropout(A)
18:
19: C ← A @ V
20: C ← C.view(B, L, H · d_h)
21:
22: if is_nested: Y ← C
23: else:        Y ← C W_O^T + (C A_O^T) B_O^T + b_O
24:
25: return Y
```

## 4.7 Complexity Analysis

### 4.7.1 Time Complexity

| Phase | Time |
|---|---|
| Q/K/V projection | $O(L d^2)$ |
| Attention scores | $O(L^2 d_h)$ |
| α-entmax | $O(n_\text{iter} L^2)$ |
| Context | $O(L^2 d_h)$ |
| Output projection | $O(L d^2)$ |
| **Total** | $O(L^2 d_h + n_\text{iter} L^2 + L d^2)$ |

**Compared to softmax:** AdaSplash adds $n_\text{iter} \times L^2$ operations 
for entmax bisection. For $n_\text{iter} = 25$, this is ~25× 
slower attention.

**For causal decoders with splash masking,** the mask restricts nonzero 
entries to $\sim W \cdot L$, reducing effective attention cost. In practice, 
this is not leveraged by PyTorch dense matmul.

### 4.7.2 Space Complexity

**Attention matrix:** $O(L^2)$ — same as softmax attention.

### 4.7.3 Parameter Count

| Config | Parameters |
|---|---|
| Original | $4 d^2$ per layer |
| AdaSplash + LoRA($r_L$) | $4 d^2 + 4 r_L (d + d)$ |
| **Change** | **+$8 r_L d$** |

For $d = 768$, $r_L = 16$: +98,304 params per layer. Across 6 layers 
(DistilBERT): +589,824 params.

## 4.8 Theoretical Notes

### 4.8.1 α-Entmax Support Sparsity

**Claim.** For $\alpha > 1$, the support of α-entmax output has size at most 
$L - k$ where $k$ is the number of scores below the threshold $\tau^*$.

**Proof.** Any score $s_i$ with $(\alpha - 1) s_i < \tau^*$ yields $p_i = 0$. 
The support is exactly $S = \{i : (\alpha - 1) s_i \geq \tau^*\}$. $\square$

### 4.8.2 Entmax vs Softmax on Peaked Distributions

**Claim.** For scores with high variance, α-entmax with $\alpha = 1.5$ 
produces sparser outputs than softmax.

**Proof sketch.** The Tsallis entropy with $\alpha > 1$ penalizes spread 
more than Shannon entropy. When score variance is high, the argmax pushes 
mass to the top scores. $\square$

### 4.8.3 Attention Sink Preservation

**Claim.** Masking first $S$ tokens as sinks does not break causality.

**Proof.** Sink tokens have $j < S \leq i$ for all queries $i \geq S$. 
For $i < S$, causality is enforced by the lower triangular mask. $\square$

### 4.8.4 Approximation Error

We provide no bound on downstream accuracy loss. The approximation error is 
determined by $\alpha$, $W$, and the interaction with LoRA recovery.

## 4.9 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| $\alpha$ | 1.5 | Higher = sparser (up to 2.0 = sparsemax) |
| $W$ (splash radius) | 32 | Higher = more context, less sparsity |
| $S$ (sink count) | 4 | Higher = more global context |
| $r_L$ (LoRA rank) | 16 | Higher = more recovery capacity |

**Choosing $\alpha$:**

- $\alpha = 1.0$: softmax (no sparsity, baseline)
- $\alpha = 1.2$: mild sparsity
- $\alpha = 1.5$: balanced (default)
- $\alpha = 1.8$: sparse, may destabilize
- $\alpha = 2.0$: sparsemax, sparse but non-smooth

**Choosing $W$:**

- $W = 16$: very local (encoder-only)
- $W = 32$: balanced (default)
- $W = 64$: broader context
- $W = L$: no windowing (baseline attention)

**Note:** Splash masking is only applied to causal decoders. Encoders use 
full attention.
