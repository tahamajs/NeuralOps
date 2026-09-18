<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real transformers
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Class | Nested? | RoPE? | GQA? | Verified |
|---|---|---|---|---|---|
| DistilBERT | `MultiHeadSelfAttention` | No | No | No | ✅ |
| RoBERTa | `BertSelfAttention` | Yes | No | No | 0.85 |
| GPT-2 | `GPT2Attention` (Conv1D) | No | No | No | 0.85 |
| Qwen3.5 | `Qwen3Attention` | No | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

The `copy_weights` method handles three extraction patterns:

### 5.2.1 GPT-2 (Fused Conv1D)

```python
if hasattr(src, "c_attn"):
    c_w = src.c_attn.weight.data.clone()
    if src.c_attn.__class__.__name__ == "Conv1D":
        c_w = c_w.t()
    q_w, k_w, v_w = c_w.chunk(3, dim=0)
```

### 5.2.2 RoBERTa (Nested)

We unwrap the nested structure:

```python
elif hasattr(src, "self"):
    s = src
    while hasattr(s, "self") and not any(
        hasattr(s, attr) for attr in ["query", "key", "value"]
    ):
        s = s.self
    # Now s is BertSelfAttention with query/key/value
```

The output projection is taken from `src.output.dense`.

### 5.2.3 DistilBERT (Flat)

Direct extraction:

```python
extracted["q"] = src.q_lin.weight.data
extracted["k"] = src.k_lin.weight.data
extracted["v"] = src.v_lin.weight.data
extracted["o"] = src.out_lin.weight.data
```

## 5.3 Nested Encoder Bypass

For RoBERTa, we set `is_nested_encoder = True` and skip our own `out_proj`:

```python
if self.is_nested_encoder:
    return context, None
output = self.out_proj(context)
```

The outer `BertSelfOutput.dense` applies the projection.

## 5.4 GQA Support

For Qwen with $H_{kv} < H$:

```python
if self.num_heads != self.num_kv_heads:
    K = repeat_kv(K, self.num_heads // self.num_kv_heads)
    V = repeat_kv(V, self.num_heads // self.num_kv_heads)
```

This is applied **before** the joint feature computation.

## 5.5 RoPE Support

Partial RoPE supported via `apply_rotary_pos_emb`.

## 5.6 Injection Algorithm

**Algorithm 3:** Inject Routing Attention
```
Input:  layer ℓ, config, old_attn
Output: new attention module

1: new ← RoutingAttention(config, pretrained_attn=old_attn)
2: if layer.attention.self: 
3:     layer.attention.self ← new; new.is_nested_encoder ← True
4: elif layer.self_attn: layer.self_attn ← new
5: elif layer.attn: layer.attn ← new
6: elif layer.self: layer.self ← new; new.is_nested_encoder ← True
7: return new
```

## 5.7 Failure Handling

| Failure | Detection | Response |
|---|---|---|
| Missing weight | `copy_weights` sets `(None, None)` | Random init |
| Shape mismatch | Runtime check | Truncate to min |
| Zero norm | `RuntimeError` raised | Injection aborts |
| Injection exception | try/except in `inject_routing` | Return `None` |

## 5.8 `RuntimeError` on Zero Norm

Unlike other modules, this one **raises** if a weight has near-zero norm:

```python
w_norm = float(torch.norm(target_mod.weight.data).item())
if w_norm < 1e-4:
    raise RuntimeError(f"RoutingAttention weight verification failed for '{proj_key}': "
                       f"norm too small ({w_norm:.2e})")
```

**This is a bug.** If any projection weight is legitimately small, the 
injection fails for the entire model. Consider replacing with a warning.

## 5.9 Double-Injection Guard

The `inject_routing` function does **not** check for prior injection. Running 
it twice could corrupt the model.

## 5.10 Diagnostics

```python
# Verify injection
for name, mod in model.named_modules():
    if 'routing' in type(mod).__name__.lower():
        print(name, type(mod).__name__)
```

## 5.11 Common Pitfalls

### 5.11.1 Cluster IDs on Padded Sequences

Padding tokens are included in k-means clustering, which may distort 
centroids. Ideally, exclude padding before clustering.

### 5.11.2 Token-0 Anchor on Long Sequences

If token 0 is itself a padding token (rare), the anchor becomes meaningless. 
Consider using the first valid token instead.

### 5.11.3 Nested Encoder Double-Projection

If `is_nested_encoder` is wrong, the output projection is applied twice.

---