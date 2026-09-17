# 4. Method

## 4.1 Overview

The wrapper `FlashAttention3` replaces a pretrained attention module and 
provides three execution paths:

1. **FlashAttention-3 kernel** (Hopper + library installed).
2. **PyTorch SDPA** (any CUDA GPU).
3. **Head mask + SDPA** (for head pruning experiments).

**Figure 4.1** shows the dispatch logic.

```
    ┌──────────────────────────────────┐
    │  Input X                          │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Q/K/V projection (frozen + LoRA)│
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Apply RoPE (if applicable)       │
    │  Apply GQA repeat_kv (if needed)  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Is FA3 available AND CUDA?       │
    └──────┬─────────────────┬──────────┘
           │ Yes             │ No
           ▼                 ▼
    ┌─────────────┐   ┌─────────────────┐
    │ flash_attn  │   │ SDPA fallback   │
    │ _func call  │   │ (F.scaled_...)  │
    └──────┬──────┘   └────────┬────────┘
           │                   │
           └─────────┬─────────┘
                     ▼
              ┌─────────────┐
              │ out_proj    │
              │ (frozen+LoRA)│
              └──────┬──────┘
                     │
                     ▼
                    Y
```

## 4.2 Phase 1: Weight Extraction

For each projection $P \in \{Q, K, V, O\}$, extract $W_P$ from the source:

- **DistilBERT:** `q_lin`, `k_lin`, `v_lin`, `out_lin`.
- **RoBERTa:** `self.query`, `self.key`, `self.value`; `output.dense` for `o`.
- **GPT-2:** Split `c_attn.weight.T`; `c_proj.weight.T` for `o`.
- **Qwen:** `q_proj`, `k_proj`, `v_proj`, `o_proj`.

Frobenius norms verified $\geq 10^{-4}$.

## 4.3 Phase 2: Frozen Base + LoRA

Each projection becomes a `LoRALinear`:

$$
\text{Proj}_P(X) = X W_P^\top + \frac{\alpha}{r_L} (X A_P^\top) B_P^\top + b_P
\tag{4.1}
$$

- $W_P$: frozen (copied from source).
- $(A_P, B_P)$: LoRA factors, $B = 0$ at init.

**Step-0 preservation.** With $B = 0$, the initial output equals the base 
output.

## 4.4 Phase 3: Position and Head Expansion

### RoPE

Applied after projection:

$$
Q, K \leftarrow \text{RoPE}(Q, K, \text{cos}, \text{sin})
\tag{4.2}
$$

Supports partial RoPE (rotary_dim < head_dim) via coordinate slicing.

### GQA

If $H_{kv} < H$:

$$
K \leftarrow \text{repeat\_kv}(K, H/H_{kv}), \quad
V \leftarrow \text{repeat\_kv}(V, H/H_{kv})
\tag{4.3}
$$

## 4.5 Phase 4: Attention Kernel Dispatch

### Path A: FlashAttention-3

If `FLASH_ATTN_AVAILABLE` and `hidden_states.is_cuda`:

```python
q_fa = q.transpose(1, 2).contiguous()  # (B, L, H, d_h)
k_fa = k.transpose(1, 2).contiguous()
v_fa = v.transpose(1, 2).contiguous()

if use_fp8 and hasattr(torch, "float8_e4m3fn"):
    q_fa = q_fa.to(torch.float8_e4m3fn)
    k_fa = k_fa.to(torch.float8_e4m3fn)
    v_fa = v_fa.to(torch.float8_e4m3fn)

out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag).to(dtype)
```

**FP8.** If `use_fp8=True` and Hopper, casts Q/K/V to FP8.

### Path B: SDPA Fallback

If FlashAttention-3 fails or is unavailable:

```python
if causal_flag and attn_mask is None:
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

**Advantage.** SDPA is always available on PyTorch 2.0+ with CUDA.

## 4.6 Phase 5: Output Projection

$$
Y = \text{Attn}(Q, K, V) W_O^\top + b_O
\tag{4.4}
$$

**Nested bypass.** For RoBERTa/BERT, `out_proj` is not applied; the outer 
`BertSelfOutput.dense` performs it.

## 4.7 The Full Algorithm

**Algorithm 1:** FlashAttention-3 wrapper forward
```
Input:  X ∈ R^{B×L×d}, attention_mask, position_embeddings
Output: Y, None

1:  Q ← LoRALinear_Q(X)
2:  K ← LoRALinear_K(X)
3:  V ← LoRALinear_V(X)
4:
5:  if RoPE: (Q, K) ← apply_rope(Q, K, cos, sin)
6:  if GQA: K, V ← repeat_kv(K, V, H/H_kv)
7:
8:  causal_flag ← is_decoder AND not cross_attention AND L == kv_len
9:
10: if FA3_AVAILABLE and is_cuda:
11:     try:
12:         if use_fp8 and fp8_supported:
13:             Q, K, V ← cast_to_fp8(Q, K, V)
14:         Y_ctx ← flash_attn_func(Q, K, V, causal=causal_flag)
15:     except:
16:         Y_ctx ← None
17: else:
18:     Y_ctx ← None
19:
20: if Y_ctx is None:                                ▷ Fallback
21:     attn_mask ← build_mask(attention_mask, causal_flag)
22:     Y_ctx ← SDPA(Q, K, V, attn_mask)
23:
24: Y_ctx ← Y_ctx.reshape(B, L, H·d_h)
25: if nested: Y ← Y_ctx
26: else: Y ← LoRALinear_O(Y_ctx)
27:
28: return Y
```

## 4.8 Complexity Analysis

| Path | Time | Memory |
|---|---|---|
| FA3 | $\mathcal{O}(L^2 d_h)$ + HBM savings | $\mathcal{O}(L d_h)$ |
| SDPA (mem-efficient) | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L d_h)$ |
| SDPA (math) | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2)$ |

**Actual time savings** depend on hardware. On Hopper, FA3 gives 2-4× over 
naive; on Ampere/Ada, SDPA memory-efficient path gives 1.5-2×.

## 4.9 Parameter Analysis

| Component | Baseline | Wrapper |
|---|---|---|
| Base weights | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA ($r_L$) | 0 | $8 r_L d$ |
| **Total per layer** | $4 d^2$ | $4 d^2 + 8 r_L d$ |

For DistilBERT ($d=768$, $r_L=16$): **+589,824 params** across 6 layers.

**Wrapper increases parameter count by ~66.4M%.**

## 4.10 Theoretical Notes

### 4.10.1 Exactness

**Claim.** FlashAttention produces the same output as naive attention up to 
floating-point precision.

**Proof.** FlashAttention uses online softmax, which is mathematically 
equivalent to standard softmax:

$$
\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}} = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}
\tag{4.5}
$$

for any shift $m$. The online variant updates $m$ incrementally and rescales 
previous accumulations, giving the same final result. $\square$

### 4.10.2 FP8 Accuracy Loss

**Claim.** FP8 attention introduces small numerical error.

**Analysis.** FP8 e4m3 has 3 mantissa bits (~6% precision). Casting Q, K, V 
to FP8 introduces error $\epsilon \sim 0.06$ per element, which can propagate 
through softmax. Empirically, we observe a small accuracy drop (see Section 8).

## 4.11 Hyperparameter Guidance

| Hyperparameter | Default | Effect |
|---|---|---|
| `use_fp8` | False | Enable FP8 on Hopper |
| `lora_rank` | 16 | Adapter capacity |
| `is_decoder` | auto | Causal masking |
| `is_nested_encoder` | auto | Output projection bypass |

---