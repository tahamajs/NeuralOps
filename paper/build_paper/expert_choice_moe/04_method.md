# 4. Method

## 4.1 Overview

ExpertChoiceMoEAttention computes standard attention, then augments it with 
an **expert-choice MoE block** on the attention output:

$$
\text{attn\_out} = \text{Attention}(Q, K, V)
\tag{4.1}
$$

$$
\text{MoE\_out} = \text{ExpertChoiceMoE}(\text{attn\_out})
\tag{4.2}
$$

$$
\text{enriched} = \text{attn\_out} + \gamma \cdot \text{MoE\_out}
\tag{4.3}
$$

$$
\text{final} = W_O \cdot \text{enriched}
\tag{4.4}
$$

where $\gamma = \sigma(b_g)$.

**Figure 4.1** shows the pipeline.

```
    Input X
       │
   ┌───┴────┐
   ▼        ▼
   Q,K,V   (identity)
   │        │
   ▼        │
   Attention │
   │        │
   ▼        │
   attn_out │
   │        │
   ├────────┤
   │        │
   ▼        │
   MoE      │
   │        │
   ▼        ▼
   enriched = attn_out + γ·MoE_out
       │
       ▼
    W_o
       │
       ▼
    Output Y
```

## 4.2 The `ExpertChoiceMoE` Block

For input $X \in \mathbb{R}^{B \times L \times d}$, we flatten to 
$\mathbf{x} \in \mathbb{R}^{T \times d}$ with $T = B \cdot L$.

### 4.2.1 Router

$$
\tilde{\mathbf{x}} = \text{LayerNorm}(\mathbf{x})
\tag{4.5}
$$

$$
\mathbf{r} = W_g \tilde{\mathbf{x}} \in \mathbb{R}^{T \times E}
\tag{4.6}
$$

with $W_g \in \mathbb{R}^{E \times d}$ initialized to uniform $(-0.01, 0.01)$.

### 4.2.2 Noisy Logits (Training Only)

$$
\mathbf{r} \leftarrow \mathbf{r} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
\tag{4.7}
$$

with $\sigma = 0.05$.

### 4.2.3 Softmax over Tokens per Expert

$$
\mathbf{s} = \text{Softmax}_T(\mathbf{r}) \in \mathbb{R}^{T \times E}
\tag{4.8}
$$

Note: softmax is over the **token dimension** (axis 0), not the expert 
dimension.

### 4.2.4 Expert-Choice Selection

For each expert $e$:

$$
\text{Top-}k_e = \text{TopK}\!\left(\mathbf{s}[:, e], k\right), \quad
k = \left\lceil \frac{T}{E} \cdot \text{capacity\_factor} \right\rceil
\tag{4.9}
$$

Every expert selects exactly $k$ tokens.

### 4.2.5 Expert Forward

$$
\mathbf{y}_e = \text{Expert}_e\!\left(\mathbf{x}[\text{Top-}k_e]\right)
\tag{4.10}
$$

Each expert is a 2-layer MLP:

$$
\text{Expert}_e(\mathbf{x}) = W_2^{(e)} \text{GELU}(W_1^{(e)} \mathbf{x})
\tag{4.11}
$$

### 4.2.6 Aggregation

$$
\mathbf{y}_{\text{out}} = \sum_e \mathbf{s}[\text{Top-}k_e, e] \cdot \mathbf{y}_e
\tag{4.12}
$$

via `index_add_` to handle overlapping tokens.

## 4.3 Zero-Init Stability

### 4.3.1 Expert Output Zero-Init

```python
for expert in self.experts:
    nn.init.zeros_(expert[2].weight)
    nn.init.zeros_(expert[2].bias)
```

**Consequence.** $\text{MoE\_out} = 0$ for any input at initialization.

### 4.3.2 Negative Gate Bias

$$
\gamma = \sigma(0.1) \approx 0.8
\tag{4.13}
$$

### 4.3.3 Effective Output at Step 0

$$
\text{enriched} = \text{attn\_out} + 0.8 \cdot 0 = \text{attn\_out}
\tag{4.14}
$$

So the augmented attention is exactly the base.

## 4.4 Perfect Load Balancing

**Claim.** With expert-choice routing and capacity factor 1.0, every expert 
processes exactly $k = T / E$ tokens.

**Proof.** Each expert selects the top-$k$ tokens by its own score, 
independently. By construction, every expert processes exactly $k$ tokens. 
$\square$

**Note.** With `capacity_factor > 1`, each expert processes $\lceil T \cdot 
\text{cf} / E \rceil$ tokens, still balanced.

## 4.5 Dynamic Per-Token Capacity

Tokens that match many experts' preferences get selected by many experts 
and receive more capacity. Tokens that match few experts get less. This 
differs from token-choice routing where every token gets exactly $k$ 
experts.

## 4.6 Cross-Attention Support

The MoE block operates on the attention output, which is computed from 
either self- or cross-attention:

```python
is_cross_attention = encoder_hidden_states is not None
kv_input = encoder_hidden_states if is_cross_attention else hidden_states
```

The MoE applies uniformly to both.

## 4.7 Algorithm

**Algorithm 1:** ExpertChoiceMoEAttention forward pass
```
Input:  X, attention_mask, position_embeddings (optional)
        encoder_hidden_states (optional)
Output: Y

 1: if cross: kv_input ← encoder_hidden_states
 2: else: kv_input ← X
 3: Q ← W_q X
 4: K ← W_k kv_input
 5: V ← W_v kv_input
 6: if RoPE: Q, K ← rotary(Q, K)
 7: if GQA: K, V ← repeat_kv(...)
 8: scores ← Q K^T / √d_h
 9: if causal: scores += causal_mask
10: scores = mask(scores, attention_mask)
11: attn_out ← softmax(scores) V
12:
13: # MoE
14: x_flat ← attn_out.view(-1, all_head_size)
15: logits ← router(x_flat)
16: scores ← softmax(logits, dim=0)   ▷ over tokens
17: for e in 0..E-1:
18:     topk_vals, topk_idx ← TopK(scores[:, e], k)
19:     expert_out ← Expert_e(x_flat[topk_idx])
20:     out_flat.index_add_(0, topk_idx, expert_out * topk_vals)
21: MoE_out ← out_flat.view(B, L, all_head_size)
22:
23: γ ← σ(b_g)
24: enriched ← attn_out + γ · MoE_out
25: Y ← W_o enriched
26: return Y
```

## 4.8 Complexity Analysis

| Component | Time | Params |
|---|---|---|
| Attention | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(d^2)$ |
| Router | $\mathcal{O}(T \cdot E \cdot d)$ | $E \cdot d$ |
| TopK per expert | $\mathcal{O}(T \log T \cdot E)$ | 0 |
| Expert forward | $\mathcal{O}(T \cdot d^2)$ | $E \cdot 2d^2$ (see below) |
| Gate | $\mathcal{O}(1)$ | 1 |

## 4.9 Parameter Analysis

**ExpertChoiceMoEAttention increases parameter count.**

| Component | Baseline | With MoE |
|---|---|---|
| Q, K, V, O | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA | 0 | $8 r_L d$ |
| Router | 0 | $E d$ |
| Experts ($E$) | 0 | $E \cdot (d \cdot 2d + 2d \cdot d)$ |
| Gate | 0 | 1 |

For $d = 768$, $E = 8$:

| Component | Params |
|---|---|
| LoRA | ~98,304 |
| Router | ~6,144 |
| Experts | ~$8 \cdot 4 d^2 = 18.9$M |
| Gate | 1 |
| **Total added** | **~19M per layer** |

Relative to baseline attention ($4 d^2 = 2.36$M):

$$
\text{increase} \approx \frac{19}{2.36} \approx 800\%
\tag{4.15}
$$

## 4.10 Training

Trainable parameters:

- LoRA on Q, K, V, O.
- MoE router + experts.
- Gate bias $b_g$.

Frozen:

- Base Q, K, V, O projections.

**Auxiliary loss.** Unlike token-choice MoE, we do **not** need an 
auxiliary load-balancing loss because expert-choice routing is guaranteed 
balanced. We do, however, track a **usage entropy** for monitoring.

```python
# Monitoring only, not optimized
usage = torch.bincount(selected_indices, minlength=E).float()
entropy = -(usage / usage.sum() * torch.log(usage / usage.sum() + 1e-9)).sum()
```

---