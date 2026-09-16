# 4. Method

## 4.1 Overview

MoE-DeltaKV-Paged has four stages:

1. **Projection + RoPE** (§4.2): standard Q/K/V projection with optional RoPE.
2. **Delta-KV compression** (§4.3): temporal deltas → SVD → 4-bit quantization → reconstruction.
3. **Paged attention with MoBA routing** (§4.4): top-$k$ page selection per query page.
4. **Gated fusion + output projection** (§4.5): mixture weights × page contexts + output.

```
        Input X
           │
           ▼
    ┌──────────────────────┐
    │ Q/K/V projection     │
    │ + RoPE               │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Delta-KV compression │
    │   ΔK = K - shift(K)  │
    │   SVD → 4-bit quant  │
    │   cumsum reconstruct │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Page partition       │
    │   P tokens per page  │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ MoBA Router          │
    │   ℓ_ij = q̄_i · k̄_j    │
    │   TopK pages         │
    │   Page-0 boost       │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Block attention      │
    │   over selected pages│
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ Gated fusion         │
    │   Σ g_j · out_j      │
    └───────┬──────────────┘
            │
    ┌───────▼──────────────┐
    │ out_proj             │
    └───────┬──────────────┘
            │
        Output
```

## 4.2 Projection + RoPE

Standard LoRALinear projections:

```python
q = self.query(hidden_states)
k = self.key(hidden_states)
v = self.value(hidden_states)
```

Reshape to $[B, H, N, d_h]$ and apply RoPE if `position_embeddings` is provided.

## 4.3 Delta-KV Compression

### 4.3.1 Delta Computation

```python
delta_K = K[:, :, 1:, :] - K[:, :, :-1, :]  # [B, H, N-1, d]
delta_V = V[:, :, 1:, :] - V[:, :, :-1, :]
```

### 4.3.2 Low-Rank SVD + 4-bit Quantization

```python
def low_rank_svd_quantize(delta, rank):
    B, H, N, d = delta.shape
    delta_flat = delta.reshape(-1, d)
    
    U, S, Vh = torch.linalg.svd(delta_flat.float(), full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
    
    delta_approx = ((U * S.unsqueeze(0)) @ Vh).to(delta.dtype)
    delta_approx = delta_approx.reshape(B, H, N, d)
    
    return ste_quantize_4bit(delta_approx)
```

**Why SVD first?** Temporal deltas have low-rank structure (natural language is 
smooth), so truncating to rank $r$ removes noise without losing signal. The 
4-bit quantization then simulates low-precision storage.

### 4.3.3 Reconstruction

```python
K_hat[:, :, 0] = K[:, :, 0]
K_hat[:, :, 1:] = K[:, :, 0:1] + torch.cumsum(delta_K_hat, dim=2)
```

### 4.3.4 Caveat: This is Simulation

The output $K_\text{hat}$ has the **same shape** as $K$. No memory is 
saved. The delta + SVD + quantization pipeline is a **training-time 
regularization** that makes the model robust to future real compression.

## 4.4 MoBA Router

### 4.4.1 Page Partition

Partition $q, k_\text{hat}, v_\text{hat}$ into $N_p = \lceil N / P \rceil$ 
pages of size $P$ (padding with zeros if needed):

```python
num_pages = (N + P - 1) // P
pad_len = num_pages * P - N
q_padded = F.pad(q, (0, 0, 0, pad_len))
k_padded = F.pad(k_hat, (0, 0, 0, pad_len))
```

### 4.4.2 Page-Mean Projections

```python
q_bar = (q_blocks * valid_b.unsqueeze(-1)).sum(dim=3) / valid_b.sum(dim=3, keepdim=True)
k_bar = (k_blocks * valid_b.unsqueeze(-1)).sum(dim=3) / valid_b.sum(dim=3, keepdim=True)
```

### 4.4.3 Router Logits

$$
\ell_{ij} = \frac{(W_\text{rg} \bar{q}_i) \cdot \bar{k}_j}{\sqrt{d_h}}
\tag{4.1}
$$

$W_\text{rg}$ is initialized to identity.

### 4.4.4 Causal Masking

For decoders:

$$
\ell_{ij} \leftarrow -\infty \quad \text{if } j > i
\tag{4.2}
$$

### 4.4.5 CLS Boost

$$
\ell_{i, 0} \leftarrow \ell_{i, 0} + 10^4
\tag{4.3}
$$

### 4.4.6 Top-K Selection

```python
k_experts = min(self.k_experts, num_pages)
_, topk_indices = torch.topk(router_logits_biased, k_experts, dim=-1)
topk_logits = torch.gather(router_logits, -1, topk_indices)
g_j = F.softmax(topk_logits, dim=-1)
```

## 4.5 Block Attention

### 4.5.1 Gather Selected Pages

```python
expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, P, d)
k_blocks_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
k_gathered = torch.gather(k_blocks_exp, dim=3, index=expanded_indices)
```

### 4.5.2 Attention Per Expert

```python
attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-1, -2)) / math.sqrt(d)
attn_scores = attn_scores.masked_fill(~k_valid.unsqueeze(-2), -10000.0)
attn_probs = F.softmax(attn_scores, dim=-1)
context_per_expert = torch.matmul(attn_probs, v_gathered)
```

### 4.5.3 Gated Fusion

```python
g_j_exp = g_j.unsqueeze(-1).unsqueeze(-1)
context = (context_per_expert * g_j_exp).sum(dim=3)
```

## 4.6 Full Algorithm

**Algorithm 1:** MoE-DeltaKV-Paged forward
```
Input:  X ∈ R^{B×N×d}, attention_mask, position_embeddings
Output: Y ∈ R^{B×N×d}

 1: Q ← query(X); K ← key(X); V ← value(X)
 2: if RoPE: (Q, K) ← apply_rotary(Q, K)
 3: (K_hat, V_hat) ← simulate_paged_delta_kv(K, V, rank)  ▷ Delta-KV
 4: Partition into N_p pages of size P
 5: Compute q_bar, k_bar (page means)
 6: ℓ ← (W_rg q_bar) · k_bar^T / √d_h                    ▷ Router
 7: if causal: ℓ ← mask_triu(ℓ)
 8: ℓ[:, :, :, 0] += 1e4                                   ▷ CLS boost
 9: topk_idx ← TopK(ℓ, k_experts)
10: g ← softmax(gather(ℓ, topk_idx))
11: Gather K, V from selected pages
12: scores ← q @ k_gathered^T / √d_h
13: attn ← softmax(scores)
14: context ← Σ g_j · (attn @ v_j)                        ▷ Gated fusion
15: Y ← out_proj(context)
16: return Y
```

## 4.7 Complexity Analysis

### 4.7.1 Per-Query-Page Complexity

For query page $i$:
- Router: $\mathcal{O}(N_p \cdot d_h)$
- Attention over selected pages: $\mathcal{O}(k_\text{exp} \cdot P^2 \cdot d_h)$

Total: $\mathcal{O}(N_p \cdot k_\text{exp} \cdot P^2 \cdot d_h)$.

**Compare to standard attention:** $\mathcal{O}(N^2 d_h) = \mathcal{O}(N_p^2 P^2 d_h)$.

**Speedup:** $\frac{N_p^2 P^2}{N_p \cdot k_\text{exp} \cdot P^2} = \frac{N_p}{k_\text{exp}}$.

For $N_p \gg k_\text{exp}$: substantial speedup.

### 4.7.2 Memory

Router logits: $\mathcal{O}(N_p^2)$.

Gathered K/V: $\mathcal{O}(N_p \cdot k_\text{exp} \cdot P \cdot d_h)$.

### 4.7.3 Delta-KV Cost

The Delta-KV step adds:
- Delta computation: $\mathcal{O}(N d_h)$
- SVD: $\mathcal{O}(\min((BH(N-1)) d_h^2, (BH(N-1))^2 d_h))$
- Quantization: $\mathcal{O}(N d_h)$

SVD dominates for large $N$.

## 4.8 Parameter Analysis

| Component | Params |
|---|---|
| query | $d^2$ (frozen) |
| key | $d^2$ (frozen) |
| value | $d^2$ (frozen) |
| out_proj | $d^2$ (frozen) |
| $W_\text{rg}$ | $d_h^2$ (trainable) |
| LoRA (4 proj) | $4 r_L (d + d_h)$ |
| **Total trainable** | $d_h^2 + 4 r_L (d + d_h)$ |

For $d = 768$, $d_h = 64$, $r_L = 16$:

| Component | Params |
|---|---|
| $W_\text{rg}$ | 4,096 |
| LoRA | ~98K |
| **Total** | **~102K** |

## 4.9 Theoretical Notes

### 4.9.1 Delta Sparsity

**Claim.** Natural language exhibits temporal smoothness: 
$\|\Delta k_t\| \ll \|k_t\|$.

**Evidence.** See Section 8.5 (empirical).

### 4.9.2 STE Correctness

**Claim.** STE preserves gradient flow through quantization.

**Proof.** $\frac{\partial \text{STE}(x)}{\partial x} = 1$ by construction 
(the detach stops gradient on the correction term). $\square$

### 4.9.3 CLS Preservation

**Claim.** With the $+10^4$ boost, page 0 is always in the top-$k$.

**Proof.** For any $k \geq 1$ and any logits $\ell_{ij}$ with $|\ell| < 10^3$, 
the boosted $\ell_{i, 0} > 10^3$ exceeds all others. $\square$

### 4.9.4 Causality

**Claim.** For decoders, causality is preserved.

**Proof.** Router logits mask $j > i$ to $-\infty$; page-level causal mask 
enforces block-causality. Within a page, no mask is needed because a page 
contains consecutive tokens. $\square$

---