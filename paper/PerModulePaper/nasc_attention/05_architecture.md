<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support Matrix

| Architecture | Nested? | RoPE? | GQA? | Default path |
|---|---|---|---|---|
| DistilBERT | No | No | No | **Dense** (encoder) |
| RoBERTa | Yes | No | No | **Dense** (encoder) |
| GPT-2 | No | No | No | **Cascade** (decoder) |
| Qwen3.5 | No | Yes | Yes | **Cascade** (decoder) |

## 5.2 Why Encoders Use Dense by Default

The flag `enable_cascade_on_encoders` defaults to `True` **in the config 
but the `use_dense` decision also checks**:

```python
use_dense = is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders)
```

If `enable_cascade_on_encoders=True` (default), encoders use cascade. If 
`False`, encoders use dense. **Ambiguity in the default** is a design 
concern.

## 5.3 Weight Transfer

Standard four-architecture copy:

```python
# GPT-2
c_w = src.c_attn.weight.data
if is_conv1d:
    c_w = c_w.t()
q_w, k_w, v_w = c_w.chunk(3, dim=0)
o_w = src.c_proj.weight.data.t()

# RoBERTa
q_w = src.self.query.weight.data
k_w = src.self.key.weight.data
v_w = src.self.value.weight.data
o_w = src.output.dense.weight.data

# DistilBERT
q_w = src.q_lin.weight.data
# ...
```

## 5.4 GQA Handling

```python
if proj_key in ("k", "v"):
    if src_out == out_d:
        target_w.data.copy_(w_oriented)
    elif src_out > out_d and src_out % out_d == 0:
        # Mean-pool
        rep = src_out // out_d
        target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
    elif src_out < out_d and out_d % src_out == 0:
        # Repeat
        rep = out_d // src_out
        target_w.data.copy_(w_oriented.repeat(rep, 1))
```

This handles GQA where $H_{kv} < H$ (Qwen, LLaMA).

## 5.5 Injection

**Algorithm 2:** Inject NASC
```
1: new ← NASCAttention(config, old=old_attn)
2: if copy_weights fails: raise (NOT return None)
3: if ℓ has .attention.self:
4:     ℓ.attention.self ← new; is_nested ← True
5: elif ℓ has .attention: ℓ.attention ← new
6: elif ℓ has .self: ℓ.self ← new; is_nested ← True
7: elif ℓ has .self_attn: ℓ.self_attn ← new
8: elif ℓ has .attn: ℓ.attn ← new
9: return new
```

**Important.** Unlike other modules, `inject_nasc` **raises** on failure:

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e
```

This violates the "safe injection returning None" contract.

## 5.6 Common Pitfalls

### 5.6.1 RFF Causal Path Memory

The `KV_cumsum` tensor in `_global_spectral_rff_attention` has shape 
$(B, H, L, 2F, d)$. For $L = 128$, this is manageable. For $L = 512$, it 
may OOM.

### 5.6.2 Batch-Level Gate

The gate is computed from `hidden_states.mean(dim=1)`, which averages over 
all tokens. This eliminates per-token adaptivity.

### 5.6.3 `softpick` Batch Fallback

The fallback triggers when **any** row has zero denominator (batch-level), 
not per-row:

```python
zero_mask = denom <= 1e-8  # shape (B, H, L, 1)
if zero_mask.any():
    fallback = F.softmax(x, dim=dim)
    p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
```

The `torch.where` uses per-element fallback, but the `if` only triggers 
if any element is affected. This is slightly better than fully batch-level.

### 5.6.4 The `_logged_once` Pattern

NASC logs the path once per instance:

```python
if not hasattr(self, '_logged_once'):
    logger.debug("NASCAttention: L=%d, ...")
    self._logged_once = True
```

This is a fix from previous log spam.

---