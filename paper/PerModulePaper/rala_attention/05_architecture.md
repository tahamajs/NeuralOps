<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

The `copy_weights` method handles three patterns:

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    w = src.c_attn.weight.data
    if isinstance(src.c_attn, Conv1D):
        w = w.t()  # (in, out) → (out, in)
    q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

```python
elif hasattr(src, "self") and hasattr(src.self, "query"):
    extracted["q"] = src.self.query.weight.data
    extracted["k"] = src.self.key.weight.data
    extracted["v"] = src.self.value.weight.data
    extracted["o"] = src.output.dense.weight.data
```

### 5.2.3 DistilBERT (Flat)

```python
else:
    extracted["q"] = src.q_lin.weight.data
    extracted["k"] = src.k_lin.weight.data
    extracted["v"] = src.v_lin.weight.data
    extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Bypass

For RoBERTa, `out_lin` is outside the attention. We set 
`is_nested_encoder = True` and skip `out_proj` in the forward pass.

```python
if self.is_nested_encoder:
    return out, None
output = self.out_proj(out)
```

For DistilBERT, `out_lin` is inside; `is_nested_encoder = False`.

## 5.4 GQA Support

For Qwen3.5 with $H_{kv} < H$:

```python
if self.num_heads != self.num_kv_heads:
    repeats = self.num_heads // self.num_kv_heads
    k = repeat_kv(k, repeats)
    v = repeat_kv(v, repeats)
```

## 5.5 RoPE Support

We support partial RoPE:

```python
if position_embeddings is not None:
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

`apply_rotary_pos_emb` handles cases where `rotary_dim < head_dim`.

## 5.6 Injection Algorithm

**Algorithm 2:** Inject RALA into a layer
```
Input:  layer, config, old_attn
Output: new attention module

1: new ← RALALinearAttention(config, old_attn=old_attn)
2: if copy_weights failed:
3:     log warning
4: if layer.attention.self:
5:     layer.attention.self ← new; new.is_nested_encoder ← True
6: elif layer.attention:
7:     layer.attention ← new
8: elif layer.self_attn:
9:     layer.self_attn ← new
10: return new
```

## 5.7 Double-Injection Guard

We check for prior injection:

```python
if isinstance(src_attn, RALALinearAttention):
    return None
if "RALA" in src_attn.__class__.__name__:
    return None
```

## 5.8 Qwen3.5 DeltaNet Skip

Some Qwen layers use GatedDeltaNet, which is incompatible with RALA. We 
skip these:

```python
if "DeltaNet" in c_name or "LinearAttention" in c_name:
    return None
```

## 5.9 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` logs | Random init for that proj |
| Shape mismatch | Shape check | Truncate or repeat |
| Zero norm | Norm check | Warning, continue |
| Injection exception | try/except | Return None |

## 5.10 Compatibility Notes

### 5.10.1 Cross-Attention

RALA does **not** support cross-attention. `encoder_hidden_states` is ignored.

### 5.10.2 Padded Sequences

Padding tokens are masked via `get_token_valid_mask`. The prefix scan 
respects this mask: padded positions do not contribute to $S$ or $z$.

### 5.10.3 Mixed Precision

RALA runs in the base model's dtype (FP16 typically). The prefix sums are 
computed in FP16, which may lose precision for long sequences. Consider 
FP32 for $L > 512$.

---