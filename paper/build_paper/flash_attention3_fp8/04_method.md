<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core implementation
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

Our implementation wraps FlashAttention-3 with three layers of safety:

1. **Import-time detection.** At module load, we attempt to import 
   `flash_attn_interface` or `flash_attn`. If neither is available, we set 
   `FLASH_ATTN_AVAILABLE = False`.

2. **Runtime FP8 detection.** At each forward, we check that:
   - `FLASH_ATTN_AVAILABLE` is True
   - `self.use_fp8` is True
   - `q.is_cuda` is True
   - `torch.float8_e4m3fn` is available
   
   If all hold, we cast Q/K/V to FP8 and call `flash_attn_func`.

3. **SDPA fallback.** If any check fails, we call 
   `F.scaled_dot_product_attention` instead.

```
              ┌─────────────────────────────┐
              │  Input Q, K, V              │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │  FLASH_ATTN_AVAILABLE ?     │
              └──────┬──────────────┬───────┘
                     │ No           │ Yes
                     │              │
                     │              ▼
                     │    ┌─────────────────┐
                     │    │ self.use_fp8 ?  │
                     │    └──┬─────────┬────┘
                     │       │ No      │ Yes
                     │       │         │
                     │       │         ▼
                     │       │   ┌──────────────┐
                     │       │   │ has float8? │
                     │       │   └──┬───────┬───┘
                     │       │      │ No    │ Yes
                     │       │      │       │
                     │       ▼      ▼       ▼
                     │    ┌──────────────────────┐
                     └───►│  SDPA fallback       │
                          └──────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────────┐
                          │  Output              │
                          └──────────────────────┘
```

## 4.2 Detection Logic

### 4.2.1 FlashAttention Import

```python
try:
    from flash_attn_interface import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        from flash_attn import flash_attn_func
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
```

The first import path is used by the Hopper-optimized `flash-attn-3` package; 
the second by the community `flash-attn` package.

### 4.2.2 FP8 Path Condition

At each forward, we check four conditions:

```python
use_fp8_path = (
    FLASH_ATTN_AVAILABLE
    and self.use_fp8
    and q.is_cuda
    and hasattr(torch, "float8_e4m3fn")
)
```

**Note.** We do **not** check CUDA capability directly; if the hardware 
does not support FP8, `flash_attn_func` will raise a runtime error and we 
catch it, falling back to SDPA.

## 4.3 FP8 Cast

When the FP8 path is taken:

```python
q_fa = q.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
k_fa = k.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
v_fa = v.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
out = flash_attn_func(q_fa, k_fa, v_fa, causal=causal_flag).to(dtype)
```

**Cast order.** Q and K are cast to e4m3 (higher precision for products); 
V is cast to the same. The output is returned in FP16/BF16.

**No scaling.** Unlike TransformerEngine, we do not apply per-tensor scaling. 
This is correct for well-behaved distributions but may lose precision for 
models with large activations.

## 4.4 SDPA Fallback

If the FP8 path is not taken:

```python
if causal_flag and attn_mask is None:
    out_sdpa = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=True, dropout_p=self.dropout.p
    )
else:
    # Build causal mask, add padding mask
    out_sdpa = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=self.dropout.p
    )
```

PyTorch's SDPA dispatches internally to:
- FlashAttention-2 (if installed and CUDA >= 11.8)
- Memory-efficient attention
- Native math

## 4.5 Architecture-Specific Weight Transfer

The `copy_weights` method extracts Q, K, V, and output projections from four 
families:

### 4.5.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    w = src.c_attn.weight.data
    if src.c_attn.__class__.__name__ == "Conv1D":
        w = w.t()  # (in, out) → (out, in)
    q_w, k_w, v_w = w.chunk(3, dim=0)
    extracted["q"] = (q_w, ...)
    ...
```

### 4.5.2 RoBERTa (Nested)

```python
elif hasattr(src, "self") and hasattr(src.self, "query"):
    q_w = src.self.query.weight.data
    k_w = src.self.key.weight.data
    v_w = src.self.value.weight.data
    if hasattr(src.output, "dense"):
        o_w = src.output.dense.weight.data
```

### 4.5.3 DistilBERT (Flat)

```python
else:
    q_w = src.q_lin.weight.data
    k_w = src.k_lin.weight.data
    v_w = src.v_lin.weight.data
    o_w = src.out_lin.weight.data
```

### 4.5.4 Qwen (GQA + RoPE)

Qwen uses separate `q_proj`, `k_proj`, `v_proj`, `o_proj`. Standard extraction.

## 4.6 Nested Encoder Bypass

For RoBERTa and other BERT-family models, the output projection lives 
outside the self-attention module:

```
BertAttention
├── self: BertSelfAttention
│   ├── query
│   ├── key
│   └── value
└── output: BertSelfOutput
    └── dense  ← output projection
```

We set `is_nested_encoder = True`, which causes `forward` to return raw 
`context` (without applying `out_proj`). The outer `output.dense` applies 
the projection.

For DistilBERT, `out_lin` is inside the attention module, so we set 
`is_nested_encoder = False`.

## 4.7 LoRA Adaptation

Each projection is wrapped in `LoRALinear`:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, ..., lora_rank=16):
        self.weight = nn.Parameter(...)      # frozen after copy
        self.lora_A = nn.Parameter(...)      # trainable
        self.lora_B = nn.Parameter(...)      # trainable, initialized to 0
    
    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out
```

## 4.8 Complexity Analysis

| Component | Time | Memory |
|---|---|---|
| Q/K/V projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |
| FlashAttention-3 FP8 | $\mathcal{O}(BL^2 d / S)$ | $\mathcal{O}(BLd + S^2)$ |
| SDPA (fallback) | $\mathcal{O}(BL^2 d)$ | $\mathcal{O}(BLd)$ |
| Output projection | $\mathcal{O}(BLd^2)$ | $\mathcal{O}(BLd)$ |

**FP8 path** uses ~$S^2$ SRAM per tile and reads Q/K/V once from HBM.

**SDPA path** uses memory-efficient attention with the same asymptotics but 
without the FP8 speedup.

## 4.9 Parameter Analysis

Our implementation **does not reduce parameter count**. The LoRA adapters 
add:

$$
\Delta p = r \cdot (d_\text{in} + d_\text{out})
\tag{4.1}
$$

per layer. For $r = 16$ and $d = 768$, this is 24,576 params per projection.

**Total overhead (DistilBERT):** 4 projections × 6 layers × 24,576 = 
**589,824 params** (+0.88%).

**This is not a compression method.**

## 4.10 Algorithm

**Algorithm 1:** FlashAttention3FP8 forward pass
```
Input:  X ∈ R^{B×L×d}, attention_mask
Output: Y ∈ R^{B×L×d}

 1: Q ← X W_Q;  K ← X W_K;  V ← X W_V    ▷ With LoRA
 2: if RoPE: (Q, K) ← Rotary(Q, K)
 3: if GQA:  (K, V) ← repeat_kv(K, V)
 4:
 5: if FLASH_ATTN_AVAILABLE AND use_fp8 AND q.is_cuda AND has_float8:
 6:     q8, k8, v8 ← cast_to_fp8(Q, K, V)
 7:     try:
 8:         out ← flash_attn_func(q8, k8, v8, causal)
 9:     except:
10:         out ← NULL  ▷ Fallback
11: else:
12:     out ← NULL
13:
14: if out is NULL:
15:     out ← SDPA(Q, K, V, mask, causal)
16:
17: if not is_nested_encoder:
18:     out ← out_proj(out)
19: return out
```

---