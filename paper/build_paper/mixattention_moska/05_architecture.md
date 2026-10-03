<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | Nested? | RoPE? | GQA? | FASA pruning? |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | ✅ |
| GPT-2 | `GPT2Attention` | No | No | No | ❌ (causal) |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | ❌ (causal) |

## 5.2 Weight Transfer

### 5.2.1 GPT-2 Conv1D

```python
c_w = src.c_attn.weight.data
is_conv1d = "Conv1D" in cls_name or c_w.shape[0] < c_w.shape[1]
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa Nested

```python
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data
```

### 5.2.3 DistilBERT Flat

```python
q_w = src.q_lin.weight.data
# ... etc
```

### 5.2.4 GQA Handling (Qwen)

For the K/V projections with $H_{kv} < H$:

```python
if src_out == out_d:
    target_w.data.copy_(w_oriented)
elif src_out > out_d and src_out % out_d == 0:
    rep = src_out // out_d
    target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
elif src_out < out_d and out_d % src_out == 0:
    rep = out_d // src_out
    target_w.data.copy_(w_oriented.repeat(rep, 1))
```

This handles GQA pooling (mean) or expansion (repeat) as needed.

## 5.3 Injection Algorithm

**Algorithm 2:** Inject FASA
```
Input:  layer ℓ, config, old_attn
Output: new attention in ℓ

1: new ← MixAttentionMoSKA(config, old=old_attn, fasa_keep_ratio=r)
2: if copy_weights fails: return None
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .self: ℓ.self ← new; is_nested ← True
6: elif ℓ has .self_attn: ℓ.self_attn ← new
7: elif ℓ has .attn: ℓ.attn ← new
8: return new
```

## 5.4 Masking

For the pruned path, the padding mask is gathered:

```python
pruned_mask = mask_2d.gather(1, idx)
```

For the causal path, the padding mask is applied directly to the full 
attention.

## 5.5 Nested Encoder Detection

```python
has_out = any(hasattr(src, x) for x in ["dense", "out_proj", "out_lin", "c_proj", "o_proj"])
is_nested = cls_name in ["BertSelfAttention", "RobertaSelfAttention", ...]
if hasattr(src, "self") and hasattr(src, "output"):
    is_nested = True
```

## 5.6 Registry

```python
@ATTENTION_REGISTRY.register("fasa")
@ATTENTION_REGISTRY.register("fasa_attention")
@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention): ...
```

## 5.7 Common Pitfalls

### 5.7.1 Decoder Path is a No-Op

For GPT-2, FASA does **no pruning**. Users expecting speedup should be 
aware.

### 5.7.2 Pruning Every Forward Pass

The importance score is recomputed every forward pass. This means the 
selected $\mathcal{S}$ changes between calls on the same input if batch 
composition changes. Not a bug, but worth documenting.

### 5.7.3 `attention_mask` Fallback

```python
mask = kwargs.get("mask", None)
if attention_mask is None and mask is not None:
    attention_mask = mask
```

Silent fallback. Not harmful but non-obvious.

---