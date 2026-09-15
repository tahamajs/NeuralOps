# 4. Method

## 4.1 Overview

SOA Fused Attention replaces the standard Q, K, V projections with 
factorized pairs:

- $W_Q \to (\text{q\_down}, \text{q\_up})$
- $W_K \to (\text{k\_down}, \text{k\_up})$
- $W_V \to (\text{v\_down}, \text{v\_up})$

The output projection $W_O$ is **not factorized** — it's retained as a 
LoRALinear. This is a deliberate choice: $W_O$ combines all heads and is 
critical for downstream accuracy.

**Figure 4.1** shows the pipeline.

```
    Input X ∈ R^{B×L×d}
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  q_down    k_down    v_down
  (k×d)     (k×d)     (k×d)
    │         │         │
    ▼         ▼         ▼
  q_up      k_up      v_up
  (d×k)     (d×k)     (d×k)
    │         │         │
    ▼         ▼         ▼
    Q         K         V
    │         │         │
    └─────────┼─────────┘
              │
         Attention
              │
              ▼
          out_proj
              │
              ▼
          Output Y
```

## 4.2 Factorization Details

For each of Q, K, V:

**Step 1:** Extract the pretrained weight $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$.

**Step 2:** Compute truncated SVD:

$$
W \approx U_k \Sigma_k V_k^\top
\tag{4.1}
$$

where $k$ is the target rank.

**Step 3:** Split symmetrically:

$$
\text{down.weight} = V_k^\top \in \mathbb{R}^{k \times d_\text{in}}
\tag{4.2}
$$

$$
\text{up.weight} = U_k \Sigma_k \in \mathbb{R}^{d_\text{out} \times k}
\tag{4.3}
$$

Note: we do NOT use $\Sigma^{1/2}$ on both sides; instead, we absorb $\Sigma_k$ 
entirely into the up-projection. This is equivalent mathematically but 
slightly different numerically (all $\Sigma$ in one factor).

**Step 4:** Freeze down and up base weights.

**Step 5:** Attach LoRA to up-projection for recovery.

## 4.3 Rank Selection

The rank $k$ trades off compression and accuracy:

$$
\text{Compression} = \frac{d^2}{2 k d} = \frac{d}{2k}
\tag{4.4}
$$

| $k$ | $d = 768$ ratio |
|---|---|
| 32 | 12× |
| 64 | 6× |
| 128 | 3× |
| 192 | 2× |
| 384 | 1× |

We default to $k = 128$ (3× reduction on Q/K/V, or 1.5× on the full block).

## 4.4 Output Projection Handling

$W_O$ is **not factorized**. Instead, we wrap it in a `LoRALinear`:

$$
\text{out} = W_O \cdot \text{context} + b_O + \text{LoRA}(\text{context})
\tag{4.5}
$$

**Why not factorize $W_O$?**

- $W_O$ combines attention heads — a critical mixing operation.
- Factorizing $W_O$ would compound error with Q/K/V compression.
- The parameter savings from $W_O$ alone are modest compared to Q/K/V.

## 4.5 Nested Encoder Bypass

For nested encoders (RoBERTa, BERT), the outer `attention.output.dense` 
applies the output projection. In this case, our module returns raw context 
and the outer projection handles the mixing.

```python
if self.is_nested_encoder:
    return context, None
out = self.out_proj(context)
return out, None
```

## 4.6 GQA Support

For GQA models (Qwen, LLaMA), the K and V projections have output dimension 
$H_{kv} \cdot d_h$, not $H \cdot d_h$. Our factorization handles this 
transparently:

```python
self.k_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, ...)
self.v_up = FactorizedLoRAUp(self.k_rank, self.all_kv_head_size, ...)
```

After projection, `repeat_kv` expands KV heads to match Q heads.

## 4.7 RoPE Support

RoPE is applied after projection:

```python
q = self.q_up(self.q_down(x)).view(B, N, H, d_h).permute(0, 2, 1, 3)
k = self.k_up(self.k_down(x)).view(B, N, H_kv, d_h).permute(0, 2, 1, 3)
if position_embeddings is not None:
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

The factorized Q/K weights provide the un-rotated projections, then RoPE 
applies its rotation. This composes cleanly.

## 4.8 Algorithm

**Algorithm 1:** SOA Fused Attention forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
        position_embeddings (optional)
Output: Y ∈ R^{B×L×d}

 1: Q ← q_up(q_down(X))                     ▷ Factorized
 2: K ← k_up(k_down(X))
 3: V ← v_up(v_down(X))
 4: Q, K, V ← reshape to multi-head
 5: if RoPE: Q, K ← apply_rotary(Q, K, pos)
 6: if GQA:  K, V ← repeat_kv(K, V)
 7: scores ← Q K^T / √d_h
 8: if causal: scores += causal_mask
 9: scores = mask(scores, attention_mask)
10: A = softmax(scores)
11: context ← A V
12: context ← reshape to (B, L, d)
13: if nested_encoder: return context
14: out ← out_proj(context)
15: return out
```

## 4.9 Complexity Analysis

For a single attention block:

| Component | Baseline | SOA Fused |
|---|---|---|
| Q proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d k)$ |
| K proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d k)$ |
| V proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d k)$ |
| Attention | $\mathcal{O}(L^2 d)$ | $\mathcal{O}(L^2 d)$ |
| O proj | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d^2)$ |

**Projection FLOPs reduction:** $\frac{3 d^2}{3 d k} = \frac{d}{k}$.

For $d = 768$, $k = 128$: **6× fewer projection FLOPs**.

Attention is unchanged — the quadratic term dominates at long L.

## 4.10 Parameter Analysis

For $d = 768$, $k = 128$:

| Component | Baseline | SOA Fused | Change |
|---|---|---|---|
| $W_Q$ | 589,824 | $128 \times 768 \times 2 = 196,608$ | **-67%** |
| $W_K$ | 589,824 | 196,608 | -67% |
| $W_V$ | 589,824 | 196,608 | -67% |
| $W_O$ | 589,824 | 589,824 | 0% |
| LoRA | 0 | $\sim 24,576$ | +small |
| **Total per layer** | 2,359,296 | 1,204,224 | **-49%** |

**Note.** We do not claim a 3× overall reduction — the true reduction is ~2× 
for the full attention block, because $W_O$ is not factorized.

## 4.11 Weight Transfer

Our `copy_weights` handles four architectures. For each:

1. Extract $W_Q, W_K, W_V, W_O$ from the pretrained attention.
2. For each of Q, K, V: compute SVD and initialize down/up.
3. For $W_O$: copy directly to `out_proj`.

**Failure handling:** If any projection cannot be extracted, we raise 
`RuntimeError` to prevent injecting a random-init module.

---
