<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Class | Nested? | RoPE? | GQA? | FP8 on RTX 4070? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ❌ Fallback |
| RoBERTa | `BertSelfAttention` | Yes | No | No | ❌ Fallback |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | ❌ Fallback |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | ❌ Fallback |

## 5.2 Weight Transfer Details

### 5.2.1 Conv1D Orientation (GPT-2)

HuggingFace `Conv1D` stores weight as `(in_features, out_features)`:

```python
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        self.weight = nn.Parameter(torch.empty(nx, nf))
        # nx = in_features, nf = out_features
```

Our extraction:

```python
if src.c_attn.__class__.__name__ == "Conv1D":
    w = src.c_attn.weight.data.t()  # → (nf, nx)
```

We detect Conv1D via class name **or** the shape heuristic 
`w.shape[0] < w.shape[1]` (which holds for Conv1D with $n_f > n_x$ but 
not for square matrices).

### 5.2.2 Nested vs Flat Detection

We use the following logic in `copy_weights`:

```python
if hasattr(src, "out_lin"):
    is_nested = False  # DistilBERT
elif (
    hasattr(src, "self")
    and hasattr(src.self, "query")
    and hasattr(src, "output")
    and hasattr(src.output, "dense")
):
    is_nested = True  # RoBERTa
elif cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]:
    is_nested = True
elif cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower():
    is_nested = True
else:
    is_nested = False
```

**Key insight:** check `out_lin` **first**. DistilBERT's 
`MultiHeadSelfAttention` ends with "SelfAttention" and would be misclassified 
as nested otherwise.

## 5.3 GQA Support

For Qwen3.5 with $H_{kv} = 2 < H = 8$:

```python
# In forward:
if self.num_kv_heads != self.num_heads:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

The `repeat_kv` function expands via broadcast and reshape:

```python
def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    B, H_kv, L, D = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(B, H_kv, n_rep, L, D)
    return hidden_states.reshape(B, H_kv * n_rep, L, D)
```

## 5.4 RoPE Support

We support partial RoPE, where the rotary dimension is smaller than 
`head_dim`:

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    rotary_dim = cos.shape[-1]
    if rotary_dim < q.shape[-1]:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot_embed = q_rot * cos + rotate_half(q_rot) * sin
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
    ...
```

## 5.5 Causality

For decoder models, we set `causal_flag = True` if:
- `self.is_decoder`
- AND not cross-attention
- AND $L = L_{kv}$

This ensures we only apply causal masking for self-attention, not 
cross-attention.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject FlashAttention3FP8 into a layer
```
Input:  layer ℓ (transformer block), config, old_attn
Output: new attention module in ℓ

1: new ← FlashAttention3FP8(config, old_attn=old_attn)
2: if new.copy_weights failed: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new
5:     new.is_nested_encoder ← True
6: elif ℓ has .attention:
7:     ℓ.attention ← new
8: elif ℓ has .self_attn:
9:     ℓ.self_attn ← new
10: return new
```

## 5.7 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| `flash_attn` not installed | ImportError | `FLASH_ATTN_AVAILABLE = False` |
| `float8_e4m3fn` missing | `hasattr(torch, ...)` | Skip FP8 path |
| FP8 kernel crash | try/except | SDPA fallback |
| Weight copy failure | Frobenius check | Log + skip |
| Cross-attention | `encoder_hidden_states != None` | Not supported — raise |

## 5.8 Cross-Attention

Our implementation **does not support cross-attention** in the current 
version. If `encoder_hidden_states` is provided, we raise 
`NotImplementedError`. This is acceptable because the models tested 
(DistilBERT, RoBERTa, GPT-2) are encoder-only or decoder-only.

## 5.9 Common Pitfalls

### 5.9.1 Nested Encoder Double-Projection

If `is_nested_encoder` is set incorrectly, the output projection is applied 
twice (once inside our module, once by the outer block):

```python
# WRONG for RoBERTa
if not self.is_nested_encoder:
    out = self.out_proj(out)  # would double-project
```

We verify `is_nested_encoder` by checking `hasattr(src, "out_lin")`.

### 5.9.2 Conv1D Transpose

If we forget to transpose Conv1D weights, Q/K/V shapes mismatch:

```python
# CORRECT
w = src.c_attn.weight.data.t()
```

### 5.9.3 GQA Mismatch

If `num_kv_heads` is wrong, `view(B, L, num_kv_heads, head_dim)` will fail 
or silently produce wrong results. We detect the correct value from 
`config.num_key_value_heads`.

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("flash3_fp8")
@ATTENTION_REGISTRY.register("flash_attention3_fp8")
class FlashAttention3FP8(BaseAttention): ...
```

## 5.11 Integration with LoRA Recovery

The pipeline:

1. Load base model.
2. Replace attention with `FlashAttention3FP8` via injection.
3. Freeze base weights, keep LoRA trainable.
4. Run recovery fine-tuning for 50 steps.
5. Evaluate.

This is the same pipeline as other modules in our framework.

---