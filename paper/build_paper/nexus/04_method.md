# 4. Method

## 4.1 Overview

NEXUS combines six sparse-attention mechanisms in a single forward pass:

```
        Input X
           │
           ▼
    ┌─────────────────────┐
    │ Q/K/V projections   │
    │ + RoPE              │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Block partition     │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ XAttention scoring  │
    │   top-k blocks      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Seer gating (learn) │
    │   add/override      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Dual-stage mask     │
    │   intra: dense      │
    │   inter: top-k      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Stripe mask         │
    │   (if N > 256)      │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ Softpick            │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ RFF rectification   │
    │   (every N layers)  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │ out_proj            │
    └──────────┬──────────┘
               │
        Output
```

## 4.2 Block Partitioning

```python
bs = self.cfg.block_size
nb = (N + bs - 1) // bs
```

## 4.3 XAttention Block Scoring

```python
Q_sample = Q[:, :, ::stride, :]
K_sample = K[:, :, ::stride, :]
attn_sample = torch.matmul(Q_sample, K_sample.transpose(-2, -1)) * scale

for bi in range(n_blocks):
    for bj in range(n_blocks):
        block = attn_sample[:, :, q_start:q_end, k_start:k_end]
        scores[:, :, bi, bj] = block.mean(dim=(2, 3))
```

## 4.4 Top-K Block Selection

```python
scores_inter = scores.clone()
scores_inter[:, :, diag, diag] = -inf
_, idx = scores_inter.topk(top_k, dim=-1)
mask = zeros_like(scores).scatter_(-1, idx, True)
mask[:, :, diag, diag] = True  # Always include diagonal
```

## 4.5 Seer Gating

```python
Q_pooled = Q.mean(dim=2)
K_pooled = K.mean(dim=2)
q_gate = self.q_pool(Q_pooled)  # [B, H, gate_rank]
k_gate = self.k_pool(K_pooled)
gate_score = torch.sigmoid(q_gate @ k_gate.T / T)  # [B, H]
keep_per_head = clamp(gate_score * nb, 1, nb)
```

## 4.6 Combined Mask

```python
inter_mask = inter_mask | seer_mask
inter_mask[:, :, :, 0] = True  # CLS block
inter_mask[:, :, diag, diag] = True
```

## 4.7 Stripe Mask

```python
if self.cfg.stripe_enabled and N > 256:
    stripe_group = idx // sw
    stripe_mask = (q_stripe - k_stripe).abs() <= 1
    scores = scores.masked_fill(~stripe_mask, -10000.0)
```

**Note.** For $N \leq 256$, this is inactive.

## 4.8 Softpick

```python
def softpick(x, dim=-1, temperature=1.0):
    x_scaled = torch.clamp(x.float() / temp, max=85.0)
    p = F.relu(torch.exp(x_scaled) - 1.0)
    denom = p.sum(dim=-1, keepdim=True) + 1e-8
    return (p / denom).to(orig_dtype)
```

**Concern.** For all-negative inputs, `p=0`, `denom=1e-8`, output ≈ 0. 
This can cause attention collapse in masked rows.

## 4.9 RFF Rectification

```python
Q_proj = torch.matmul(F.normalize(Q), proj).clamp(max=11.0)
K_proj = torch.matmul(F.normalize(K), proj).clamp(max=11.0)
Q_feat = torch.exp(Q_proj) / sqrt(F)
K_feat = torch.exp(K_proj) / sqrt(F)
KV = einsum("bhnf,bhnd->bhfd", K_feat, V)
Z = K_feat.sum(dim=2)
ctx = einsum("bhnf,bhfd->bhnd", Q_feat, KV)
norm = einsum("bhnf,bhf->bhn", Q_feat, Z)
return ctx / (norm + 1e-5)
```

**Note.** Only active when `layer_idx % rectify_interval == 0` AND (`pretrained_attn is None` OR `enable_spectraformer=True`).

## 4.10 Full Algorithm

**Algorithm 1:** NEXUS forward
```
Input:  X, attention_mask, position_embeddings, is_decoder
Output: Y

 1: Q, K, V ← projections(X)
 2: if RoPE: (Q, K) ← apply_rotary(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: token_valid ← extract_valid_mask(attention_mask)
 5: use_dense ← is_causal OR is_cross OR N ≤ 8
 6: if use_dense:
 7:     return standard_attention(Q, K, V)
 8: if is_rectify and spec_rectify:
 9:     return spec_rectify(Q, K, V)
10: block_scores ← XAttention_score(Q, K)
11: inter_mask ← topk(block_scores)
12: if seer_gate: seer_mask ← seer_gate(Q, K)
13: inter_mask ← inter_mask | seer_mask
14: inter_mask[:, :, :, 0] ← True
15: inter_mask[:, :, diag, diag] ← True
16: scores ← QK^T / √d_h
17: scores ← expand_mask(inter_mask, scores)
18: if N > 256: scores ← stripe_mask(scores)
19: scores ← padding_mask(scores, token_valid)
20: attn ← softpick(scores)
21: context ← attn @ V
22: context ← mask_padding(context)
23: return out_proj(context)
```

## 4.11 Complexity Analysis

### 4.11.1 Block-Sparse Complexity

- XAttention scoring: $\mathcal{O}((N/\sigma)^2 d_h)$
- Top-K: $\mathcal{O}(M^2 \log k)$
- Block attention: $\mathcal{O}(N \cdot k \cdot B_s \cdot d_h)$
- **Total:** $\mathcal{O}(N k B_s d_h)$

For $k B_s \ll N$: **sub-quadratic**.

### 4.11.2 Dense Fallback

For causal, cross, or short sequences: $\mathcal{O}(N^2 d_h)$.

## 4.12 Parameter Analysis

| Component | Params |
|---|---|
| Q/K/V/O (frozen) | $4 d^2$ |
| LoRA (4 proj) | $4 r_L (d + d_h)$ |
| Seer gate | $2 \cdot (d_h \cdot g_r + g_r \cdot g_r)$ |
| RFF proj | $d_h \cdot F$ |

For $d = 768$, $d_h = 64$, $r_L = 16$, $F = 128$:

| Component | Params |
|---|---|
| LoRA | ~98K |
| Seer gate | ~2K |
| RFF | ~8K (per rectified layer) |
| **Total trainable** | **~108K per layer** |

## 4.13 Theoretical Notes

### 4.13.1 Softpick Sparsity

**Claim.** softpick produces sparsity when logits are negative.

**Proof.** For $x_i < 0$: $e^{x_i} < 1$, so $e^{x_i} - 1 < 0$, ReLU gives 0. $\square$

**Concern.** When **all** logits are negative, softpick collapses to uniform.

### 4.13.2 Causality

**Claim.** For decoders, causality is preserved via dense fallback.

**Proof.** The `use_dense` path applies a standard triangular mask. $\square$

### 4.13.3 Block-Sparse Correctness (Encoder)

For encoder, the block mask ensures each query block attends to its own 
diagonal + top-$k$ key blocks.

---