# 4. Method

## 4.1 Overview

Our FlashAttention-3 module is a **drop-in replacement** for standard 
attention with three execution paths:

1. **FA3 path** — when `flash_attn` is available and input is on CUDA.
2. **FP8 path** — when `use_fp8=True` and Hopper is available.
3. **SDPA fallback** — when neither is available.

**Figure 4.1** shows the pipeline.

```
    Input X
       │
   ┌───┼───┬───┐
   ▼   ▼   ▼   ▼
   W_q W_k W_v  (identity)
   │   │   │
   ▼   ▼   ▼
   Q   K   V
   │   │   │
   ├───┴───┤
   │       │
   ▼       │
   RoPE    │
   │       │
   ▼       ▼
   ├───────┤
   │       │
   ▼       ▼
   GQA expand
   │
   ▼
   Dispatch:
   ├── FA3 (Hopper, CUDA)
   ├── FP8 (Hopper, use_fp8=True)
   └── SDPA (fallback)
   │
   ▼
   W_o (output proj)
   │
   ▼
   Output Y
```

## 4.2 FA3 Dispatch Logic

```python
if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
    try:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        if self.use_fp8 and hasattr(torch, "float8_e4m3fn"):
            q_fa = q_fa.to(torch.float8_e4m3fn)
            k_fa = k_fa.to(torch.float8_e4m3fn)
            v_fa = v_fa.to(torch.float8_e4m3fn)
        out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag).to(dtype)
        out = out.contiguous().view(B, L, self.all_head_size)
    except Exception:
        out = None
```

Two conditions:
- `FLASH_ATTN_AVAILABLE` — the library was imported.
- `hidden_states.is_cuda` — the tensor is on GPU.

## 4.3 SDPA Fallback

When the FA3 path fails or is unavailable:

```python
if out is None:
    eff_mask = ...  # attention or encoder mask
    attn_mask = ...  # build mask
    if causal_flag and attn_mask is None:
        out_sdpa = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, ...
        )
    else:
        ...
    out = out_sdpa.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
```

PyTorch SDPA is **exact** — same mathematical output as FA3 (modulo FP 
reassociation).

## 4.4 FP8 Mode

When `use_fp8=True` and Hopper is present:

$$
Q_\text{fp8} = \text{cast}_{e4m3fn}(Q), \quad
K_\text{fp8} = \text{cast}_{e4m3fn}(K), \quad
V_\text{fp8} = \text{cast}_{e4m3fn}(V)
\tag{4.1}
$$

The FA3 kernel performs matmuls in FP8 Tensor Cores and returns FP16/BF16 
output.

**Risk.** FP8 has limited precision; without careful scaling, this can 
degrade accuracy. The FA3 kernel handles scaling internally.

## 4.5 Weight Transfer

We copy base weights from the pretrained attention via `copy_weights`:

### 4.5.1 GPT-2 Conv1D

```python
c_w = c_attn.weight.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = c_proj.weight.t()
```

### 4.5.2 RoBERTa Nested

```python
q_w = src.self.query.weight
k_w = src.self.key.weight
v_w = src.self.value.weight
o_w = src.output.dense.weight
```

### 4.5.3 Flat Modules

Direct attribute access.

## 4.6 Nested Encoder Bypass

**Important difference from other modules.** The FA3 `copy_weights` has 
this detection order:

```python
if has_out_lin:
    is_nested = False
elif (has self and output.dense):
    is_nested = True
elif ...:
    is_nested = True
```

The `has_out_lin` check comes **first**, so DistilBERT (which has `out_lin`) 
is correctly classified as **flat**, not nested.

## 4.7 Algorithm

**Algorithm 1:** FlashAttention-3 forward
```
Input:  X, attention_mask, position_embeddings (optional)
        encoder_hidden_states (optional)
Output: Y

 1: Q ← W_q X;  K ← W_k kv_input;  V ← W_v kv_input
 2: if RoPE: Q, K ← rotary(Q, K)
 3: if GQA: K, V ← repeat_kv(K, V)
 4: causal_flag ← is_decoder and not is_cross_attention and L == kv_len
 5:
 6: # Try FA3
 7: if FLASH_ATTN_AVAILABLE and on_cuda:
 8:     try:
 9:         if use_fp8: cast to e4m3fn
10:         out ← flash_attn_func(Q, K, V, causal_flag)
11:     except: out ← None
12:
13: # Fallback
14: if out is None:
15:     out ← sdpa(Q, K, V, attn_mask, causal_flag)
16:
17: # Output projection
18: if is_nested_encoder: Y ← out
19: else: Y ← W_o out
20: return Y
```

## 4.8 Complexity Analysis

| Component | Standard attention | FA3 |
|---|---|---|
| Attention scores | $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory | $\mathcal{O}(L^2 d_h)$ time, **$\mathcal{O}(L)$ memory** |
| Softmax | $\mathcal{O}(L^2)$ | $\mathcal{O}(L^2)$ |
| @V | $\mathcal{O}(L^2 d_h)$ | $\mathcal{O}(L^2 d_h)$ |
| **Peak memory** | $\mathcal{O}(L^2)$ | **$\mathcal{O}(L)$** |

**Time** is asymptotically the same; FA3 achieves **constant-factor** 
speedup via better SRAM utilization.

**Memory** is the real win: peak drops from $\mathcal{O}(L^2)$ to 
$\mathcal{O}(L)$.

## 4.9 Parameter Analysis

**FlashAttention does NOT reduce parameter count.**

| Component | Baseline | FA3 |
|---|---|---|
| Q, K, V, O | $4 d^2$ | $4 d^2$ (frozen) |
| LoRA | 0 | $8 r_L d$ |
| **Total** | $4 d^2$ | $4 d^2 + 8 r_L d$ |

For $d = 768$, $r_L = 16$:

| Component | Params |
|---|---|
| Baseline | 2,359,296 |
| LoRA | 98,304 |
| **Total** | 2,457,600 |
| **Increase** | **+4.2%** |

The increase is only LoRA. **No compression.**

## 4.10 Training

Trainable parameters:

- LoRA adapters on Q, K, V, O.

Frozen:

- Base projections.

**Recovery fine-tuning.** Since FA3 produces exact attention, the 
recovery step is mainly for the LoRA adapters (which are zero at init). In 
practice, the recovery loss starts close to baseline loss because the 
attention output is identical to baseline.

---
