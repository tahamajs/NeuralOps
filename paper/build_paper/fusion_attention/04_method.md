# 4. Method

## 4.1 Overview

FUSIONATTN has two execution paths selected by sequence length:

- **Short path ($L \leq L_\text{threshold}$):** Standard softmax attention.
- **Long path ($L > L_\text{threshold}$):** MoBA-style block-sparse attention.

**Figure 4.1** shows the pipeline.

```
    ┌──────────────────────────────────────┐
    │  Input X → Q/K/V projections         │
    │  (frozen base + LoRA)                │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Apply RoPE, repeat_kv (GQA)         │
    └────────────────┬─────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────┐
    │  Is L ≤ 128?                          │
    └────┬──────────────────────┬──────────┘
         │ Yes                  │ No
         ▼                      ▼
    ┌────────────┐      ┌────────────────────┐
    │  Full      │      │  Block partition   │
    │  softmax   │      │  + Gate scoring    │
    │  attention │      │  + Top-K selection │
    └─────┬──────┘      │  + Block attention │
          │             └─────────┬──────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌────────────────┐
              │  out_proj      │
              │  (frozen + LoRA)│
              └────────┬───────┘
                       │
                       ▼
                      Y
```

## 4.2 Phase 1: Projections

Frozen base + LoRA:

$$
\text{Proj}_P(X) = X W_P^\top + \frac{\alpha}{r_L} (X A_P^\top) B_P^\top \cdot \sigma + b_P
\tag{4.1}
$$

where $W_P$ is copied exactly from the pretrained attention.

## 4.3 Phase 2: Length-Based Dispatch

$$
\text{Path}(L) = \begin{cases}
\text{short}, & L \leq L_\text{threshold} \\
\text{long}, & L > L_\text{threshold}
\end{cases}
\tag{4.2}
$$

with $L_\text{threshold} = 128$ (default). Also uses short path for 
cross-attention.

## 4.4 Phase 3 (Long Path): Block Gating

### Block Partitioning

Pad sequence to multiple of $B$ (block size), then reshape:

$$
Q \to Q_\text{blocks} \in \mathbb{R}^{B \times H \times T \times B \times d_h}
\tag{4.3}
$$

### Block-Mean Representations

$$
\bar{Q}_i = \frac{1}{B} \sum_{j=1}^{B} Q_\text{blocks}[:, :, i, j, :], \quad
\bar{K}_i = \frac{1}{B} \sum_{j=1}^{B} K_\text{blocks}[:, :, i, j, :]
\tag{4.4}
$$

### Gate MLP

A 2-layer MLP with GELU activation:

$$
\phi(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
\tag{4.5}
$$

with $W_1 \in \mathbb{R}^{4 d_h \times d_h}$, $W_2 \in \mathbb{R}^{1 \times 4 d_h}$.

### Gate Scores

$$
g(i, j) = \frac{\phi(\bar{Q}_i) \cdot \phi(\bar{K}_j)}{\tau}
\tag{4.6}
$$

where $\tau$ is a learnable temperature.

### Top-K Selection

$$
\text{TopK}_i = \text{top-}K_{j}(\text{softmax}_j(g(i, j)))
\tag{4.7}
$$

with block 0 always included (attention sink).

## 4.5 Phase 4 (Long Path): Block Attention

For each query block $i$:

$$
\text{Out}_i = \text{softmax}\!\left(\frac{Q_i K_{\text{TopK}_i}^\top}{\sqrt{d_h}}\right) V_{\text{TopK}_i}
\tag{4.8}
$$

with causal masking if decoder, and padding masking.

## 4.6 The Full Algorithm

**Algorithm 1:** FUSIONATTN forward
```
Input:  X, attention_mask, position_embeddings
Output: Y, attn_weights

1:  Q ← LoRALinear_Q(X)
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rope(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V)
7:
8:  if L ≤ 128 or cross_attention:                  ▷ SHORT PATH
9:      S ← QK^T / √d_h
10:     S ← causal_mask(S) if decoder
11:     S ← padding_mask(S, attention_mask)
12:     A ← softmax(S)
13:     Y_ctx ← A @ V
14:
15: else:                                              ▷ LONG PATH
16:     Q_blocks ← partition(Q, block_size)
17:     K_blocks ← partition(K, block_size)
18:     Q_means ← mean(Q_blocks, dim=-2)
19:     K_means ← mean(K_blocks, dim=-2)
20:     g ← GateMLP(Q_means) @ GateMLP(K_means)^T / τ
21:     top_idx ← topk(g, K)
22:     top_idx[..., 0] ← 0                          ▷ Always include block 0
23:     for each query block i:
24:         K_sel ← gather(K_blocks, top_idx[i])
25:         V_sel ← gather(V_blocks, top_idx[i])
26:         S ← Q_blocks[i] @ K_sel^T / √d_h
27:         S ← causal_mask + padding_mask
28:         A ← softmax(S)
29:         Out[i] ← A @ V_sel
30:     Y_ctx ← concat(Out) and crop to L
31:
32: Y_ctx ← Y_ctx.reshape(B, L, H·d_h)
33: if nested: Y ← Y_ctx
34: else: Y ← LoRALinear_O(Y_ctx)
35:
36: return Y
```

## 4.7 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| Short ($L \leq 128$) | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |
| Long ($L > 128$) | $\mathcal{O}(K \cdot L \cdot B \cdot d_h)$ | $\mathcal{O}(K \cdot B \cdot L)$ |

For $L = 512, B = 64, K = 4$:

- Short: $512^2 = 262$K
- Long: $4 \times 512 \times 64 = 131$K
- **Reduction: 2×**

For $L = 4096, B = 64, K = 4$:

- Short: $16.7$M
- Long: $1.05$M
- **Reduction: 16×**

## 4.8 Parameter Analysis

| Component | Parameters |
|---|---|
| LoRA ($r_L$) | $8 r_L d$ per layer |
| Gate MLP | $2 \times 4 d_h^2$ per layer |
| Gate temperature | 1 |
| output_scale | 1 per projection |

For DistilBERT ($d=768$, $r_L=16$, $d_h=64$):

- LoRA: $8 \times 16 \times 768 = 98{,}304$ per layer
- Gate: $2 \times 4 \times 64^2 = 32{,}768$ per layer
- Total: $131{,}072$ per layer
- **6 layers: +786,432 params**

**FUSIONATTN increases parameter count by ~66.4M%.**

## 4.9 Theoretical Notes

### 4.9.1 Short Path Exactness

**Claim.** For $L \leq L_\text{threshold}$ with $B_\text{LoRA} = 0$:

$$
\mathcal{A}'(X) = \mathcal{A}(X)
\tag{4.9}
$$

**Proof.** The short path performs standard softmax attention with copied 
base weights. With zero LoRA, output matches pretrained. $\square$

### 4.9.2 Long Path Approximation

The long path approximates full attention by restricting to top-$K$ blocks. 
Approximation error depends on gate quality. No formal bound provided.

### 4.9.3 Causality

For decoder long path, causal masking is enforced within each selected block:

$$
\text{mask}(i, j) = -\infty \text{ if } i < j
\tag{4.10}
$$

Only keys with $j \leq i$ contribute. Causality preserved.

## 4.10 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| Block size $B$ | 64 | Larger = less overhead, coarser |
| Top-K | 4 | Larger = more context, more cost |
| Threshold | 128 | Larger = more short-path |
| LoRA rank | 16 | Higher = more recovery |

---