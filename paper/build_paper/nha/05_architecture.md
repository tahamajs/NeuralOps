<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 05_architecture.md
     PURPOSE: Integration with real architectures
     DEPENDENCIES: 04_method.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 5. Architecture Integration

## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | Flat | No | No | ✅ |
| RoBERTa | Nested | No | No | ✅ |
| GPT-2 | Causal | No | No | ✅ |
| Qwen3.5 | Causal + RoPE | Yes | Yes | 0.85 |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

Direct copy of `q_lin, k_lin, v_lin, out_lin`. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_proj`.

### 5.2.3 GPT-2 (Causal)

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

Direct copy with `repeat_kv` at runtime.

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, `is_nested_encoder = True`:
- No `out_proj` applied inside NHA.
- Outer `attention.output.dense` applies output projection.

## 5.4 Causality

- **Local stream:** Band mask with $j \le i$.
- **Global stream:** Chunked prefix scan.

## 5.5 RoPE Integration

RoPE applied to Q, K **before** both streams:

```python
Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
```

## 5.6 Padding Handling

Both streams handle padding:
- **Local:** Mask invalid keys.
- **Global:** Zero out invalid K, V before prefix scan.

## 5.7 Injection Algorithm

**Algorithm 2:** Inject NHA into a full model
```
Input:  model M, config c
Output: M with NHA attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← NativeHybridAttention(config=c, pretrained_attn=src)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_nha` returns `None` on failure:

```python
try:
    new_attn = NativeHybridAttention(...)
except Exception as e:
    logger.error("NHA construction failed: %s", e)
    return None
```

## 5.9 Parameter Count

| Projection | Params |
|---|---|
| $W_Q$ | $d H d_h$ |
| $W_K$ | $d H_{kv} d_h$ |
| $W_V$ | $d H_{kv} d_h$ |
| $W_O$ | $H d_h d$ |
| Gate | $d + 1$ |
| RFF buffer | $d_h m$ (not trainable) |

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("nha")
@ATTENTION_REGISTRY.register("nha_attention")
class NativeHybridAttention(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` raises | Injection aborts |
| NaN in RFF | `torch.isnan` | Cast FP32 |
| CUDA OOM in scan | RuntimeError | Reduce chunk |
| RoPE mismatch | Exception | Fallback |

---