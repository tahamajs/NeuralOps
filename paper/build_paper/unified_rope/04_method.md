<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 04_method.md
     PURPOSE: The core method
     DEPENDENCIES: 03_background.md
     LENGTH TARGET: 3 - 4 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 4. Method

## 4.1 Overview

UnifiedRoPE implements RoPE-based scaled dot-product attention with a 
single `nn.Module`. It provides:

1. **LoRALinear projections** for Q, K, V, O.
2. **Precomputed RoPE caches** for fast frequency lookup.
3. **Robust weight transfer** across architectures.
4. **GQA expansion** via `repeat_kv`.
5. **Nested encoder bypass** for RoBERTa-style models.

**Figure 4.1** shows the pipeline.

```
   ┌──────────────────────────────────────────────┐
   │  Input X ∈ R^{B × L × d}                     │
   └────────────────────┬─────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ q_proj  │    │ k_proj  │    │ v_proj  │
   │ + LoRA  │    │ + LoRA  │    │ + LoRA  │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        ▼              ▼              │
   ┌─────────┐    ┌─────────┐         │
   │ Apply   │    │ Apply   │         │
   │ RoPE    │    │ RoPE    │         │
   └────┬────┘    └────┬────┘         │
        │              │              │
        │              ▼              │
        │         ┌─────────┐         │
        │         │ GQA     │         │
        │         │ expand  │         │
        │         └────┬────┘         │
        │              │              │
        └──────┬───────┴──────────────┘
               ▼
        ┌──────────────┐
        │  Attention   │
        │  softmax(QK) │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │  out_proj    │
        │  + LoRA      │
        └──────────────┘
```

## 4.2 The `UnifiedRoPE` Module

### 4.2.1 Initialization

```python
class UnifiedRoPE(BaseAttention):
    def __init__(self, config=None, rank=None, dim=None, num_heads=None,
                 max_seq_len=4096, base=10000, dropout=0.1,
                 pretrained_attn=None, lora_rank=16, lora_alpha=32.0,
                 is_decoder=None, **kwargs):
        # Resolve dimensions
        self.hidden_size = ...
        self.num_heads = ...
        self.num_kv_heads = ...
        self.head_dim = hidden_size // num_heads
        
        # LoRA projections
        self.q_proj = LoRALinear(hidden_size, q_head_size, lora_rank=..., freeze_base=True)
        self.k_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.v_proj = LoRALinear(hidden_size, kv_head_size, lora_rank=..., freeze_base=True)
        self.out_proj = LoRALinear(q_head_size, hidden_size, lora_rank=..., freeze_base=True)
        
        # RoPE cache
        self.rotary_dim = 2 * (head_dim // 2)
        self.register_buffer("inv_freq", ..., persistent=False)
        self._build_cache(max_seq_len)
```

### 4.2.2 Precomputed RoPE Cache

```python
def _build_cache(self, seq_len):
    t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos(), persistent=False)
    self.register_buffer("sin_cached", emb.sin(), persistent=False)
```

**Note.** Buffers are `persistent=False`, so they are not stored in 
`state_dict()`. This avoids checkpoint pollution.

### 4.2.3 Applying RoPE

```python
def _apply_rope(self, x, seq_len):
    rotary_dim = min(self.rotary_dim, self.cos_cached.shape[-1], x.shape[-1])
    rotary_dim = 2 * (rotary_dim // 2)
    
    cos = self.cos_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    sin = self.sin_cached[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(0)
    
    if rotary_dim < x.shape[-1]:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat((rotated, x_pass), dim=-1)
    else:
        return (x * cos) + (_rotate_half(x) * sin)
```

## 4.3 Weight Transfer

### 4.3.1 Five Extraction Patterns

UnifiedRoPE's `copy_weights` handles:

| Pattern | Architecture | Weights extracted |
|---|---|---|
| 1 | GPT-2 | Split `c_attn`; `c_proj` for output |
| 2 | RoBERTa | `self.query/key/value` + `output.dense` |
| 3 | Qwen | `q_proj/k_proj/v_proj/o_proj` |
| 4 | DistilBERT | `q_lin/k_lin/v_lin/out_lin` |
| 5 | Fallback | UniversalAdapter |

### 4.3.2 Conv1D Transposition

```python
if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
    c_attn_w = c_attn_w.t()
```

### 4.3.3 Nested Encoder Detection

```python
if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
    self.is_nested_encoder = True
    src = old_attn.self
    if hasattr(old_attn.output, "dense"):
        w_o = old_attn.output.dense.weight
```

When nested, `out_proj` is **skipped** in forward; the outer layer applies.

### 4.3.4 LoRA Preservation

The `_copy_layer` helper preserves source LoRA adapters:

```python
def _copy_layer(target_layer, w, b, src_sub=None):
    # Copy base weight
    if w is not None: ...
    # Copy bias
    if b is not None: ...
    # Preserve LoRA
    if src_sub is not None and hasattr(src_sub, "lora_A"):
        target_layer.lora_A.data.copy_(src_sub.lora_A.data)
        target_layer.lora_B.data.copy_(src_sub.lora_B.data)
```

## 4.4 Forward Pass

```python
def forward(self, hidden_states, attention_mask=None, ...):
    B, L, _ = hidden_states.shape
    
    # 1. Project
    q = self.q_proj(hidden_states).view(B, L, H, d_h).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, L, Hkv, d_h).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, L, Hkv, d_h).transpose(1, 2)
    
    # 2. Apply RoPE
    pos_emb = kwargs.get("position_embeddings")
    if pos_emb is not None:
        cos, sin = pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    else:
        q = self._apply_rope(q, L)
        k = self._apply_rope(k, L)
    
    # 3. GQA expansion
    if self.num_kv_heads != self.num_heads:
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
    
    # 4. Scaled dot-product attention
    scores = q @ k^T / sqrt(d_h)
    
    # 5. Causal mask for decoders
    if self.is_decoder and L > 1:
        scores = scores + causal_mask
    
    # 6. Padding mask
    if attention_mask is not None:
        scores = apply_attention_mask(scores, attention_mask)
    
    # 7. Softmax, dropout, context
    attn = softmax(scores)
    context = attn @ v
    
    # 8. Output projection (skip if nested)
    if self.is_nested_encoder:
        output = context
    else:
        output = self.out_proj(context)
    
    return output, attn
```

## 4.5 Two RoPE Entry Points

### 4.5.1 Internal `_apply_rope`

Uses the cached `cos_cached`, `sin_cached` buffers. **Fast path** for models 
that do not supply their own RoPE cache.

### 4.5.2 External `apply_rotary_pos_emb`

Accepts `(cos, sin)` from HuggingFace models that precompute their own 
RoPE. Used when `position_embeddings` is passed in `forward`.

**Why two paths?** Some models (Qwen) supply their RoPE cache; others do 
not. UnifiedRoPE supports both.

## 4.6 GQA Expansion

For $H_{kv} < H$:

```python
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)
```

`repeat_kv` broadcasts each KV head to `H / H_{kv}` query heads.

## 4.7 Nested Encoder Bypass

For RoBERTa-style models:

```python
if self.is_nested_encoder:
    output = context   # Raw context
else:
    output = self.out_proj(context)
```

The outer `BertSelfOutput.dense` applies the projection.

## 4.8 Complexity Analysis

| Phase | Time | Memory |
|---|---|---|
| Projections | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| RoPE application | $\mathcal{O}(L H d_h)$ | $\mathcal{O}(L H d_h)$ |
| GQA expansion | $\mathcal{O}(L H d_h)$ | $\mathcal{O}(L H d_h)$ |
| Attention | $\mathcal{O}(L^2 H d_h)$ | $\mathcal{O}(L^2 H)$ |
| Output projection | $\mathcal{O}(L d^2)$ | $\mathcal{O}(L d)$ |
| **Total** | $\mathcal{O}(L^2 H d_h + L d^2)$ | $\mathcal{O}(L^2 H + L d)$ |

**Same as standard attention.** No compression.

## 4.9 Theoretical Notes

### 4.9.1 RoPE Rotation Correctness

**Proposition.** The rotation operator preserves L2 norm:

$$
\|\tilde{q}_i\|_2 = \|q_i\|_2
\tag{4.1}
$$

**Proof.** Rotation matrices are orthogonal. $\square$

### 4.9.2 Relative Position Property

**Proposition.** The attention score satisfies:

$$
\tilde{q}_i^\top \tilde{k}_j = f(q_i, k_j, i - j)
\tag{4.2}
$$

**Proof.** Classical (Su et al., 2021). $\square$

### 4.9.3 GQA Equivalence

**Proposition.** GQA with `repeat_kv` produces the same output as if the 
KV heads were duplicated.

**Proof.** `repeat_kv` literally duplicates. $\square$

## 4.10 Summary

UnifiedRoPE:
1. Implements RoPE-based attention.
2. Works across four architectures.
3. Preserves base weights.
4. Supports LoRA.
5. **Does not compress.**

It is a **substrate** for other compression techniques.

---