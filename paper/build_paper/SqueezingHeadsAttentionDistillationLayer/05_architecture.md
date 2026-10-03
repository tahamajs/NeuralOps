# 5. Architecture Integration


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```


## 5.1 Model Support

| Architecture | Attention | RoPE? | GQA? | Verified |
|---|---|---|---|---|
| DistilBERT | Flat | No | No | ✅ |
| RoBERTa | Nested | No | No | ✅ |
| GPT-2 | Causal | No | No | ✅ |
| Qwen3.5 | Causal + RoPE | Yes | Yes | not tested |

## 5.2 Weight Transfer

### 5.2.1 DistilBERT (Flat)

```python
q_lin, k_lin, v_lin, out_lin
```

Direct copy with head-dim slicing. `is_nested_encoder = False`.

### 5.2.2 RoBERTa (Nested)

```python
attention.self.query, attention.self.key, attention.self.value
attention.output.dense
```

Extract from inner `self`. `is_nested_encoder = True`; skip own `out_proj`.

### 5.2.3 GPT-2 (Causal)

```python
c_attn (fused), c_proj
```

Split fused `c_attn`:

```python
w = src.c_attn.weight.t()  # Conv1D transpose
q_w, k_w, v_w = w.chunk(3, dim=0)
```

### 5.2.4 Qwen3.5 (GQA + RoPE)

GQA with pooling:

```python
if w_k.shape[0] > self.kv_head_size:
    rep = w_k.shape[0] // self.kv_head_size
    w_k_pooled = w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1)
```

## 5.3 Handling Nested Encoders

For RoBERTa/BERT, `is_nested_encoder = True`:
- No `out_proj` applied inside Squeezing Heads.
- Outer `attention.output.dense` applies output.

## 5.4 Causality

Standard causal masking in forward:

```python
causal_mask = torch.triu(torch.full((L, L), -inf), diagonal=1)
scores = scores + causal_mask
```

## 5.5 GQA Handling

For Qwen/LLaMA with $H_{kv} < H$, use `repeat_kv`:

```python
if self.num_kv_heads < self.num_heads:
    k = repeat_kv(k, self.kv_repeat)
    v = repeat_kv(v, self.kv_repeat)
```

## 5.6 Head-Dim Reduction

If the target head_dim $d_h' < d_h$, slice the leading rows of $W_Q, W_K, W_V$ 
and leading columns of $W_O$:

```python
m_q_out = min(self.q_head_size, w_q.shape[0])  # = d_h' * H
self.q_proj.weight[:m_q_out, :m_q_in].copy_(w_q[:m_q_out, :m_q_in])
```

## 5.7 Injection Algorithm

**Algorithm 3:** Inject Squeezing Heads
```
Input:  model M, config c
Output: M with LoRA-wrapped attention

1:  for ℓ in transformer_layers(M) do
2:      src ← extract_attention(ℓ)
3:      if src is None: continue
4:      new ← SqueezingHeadsDistiller(config=c, old_attn=src)
5:      replace_attention(ℓ, new)
6: end for
```

## 5.8 Failure Handling

`inject_squeezing_heads` returns `None` on failure:

```python
try:
    new_attn = SqueezingHeadsDistiller(...)
except Exception as e:
    logger.error("Failed to instantiate at layer %d: %s", layer_idx, e)
    return None
```

## 5.9 Parameter Count

| Component | Baseline | Squeezing Heads |
|---|---|---|
| Frozen $W_Q, W_K, W_V, W_O$ | $4 d^2$ | $4 d^2$ (same) |
| LoRA adapters | 0 | $4 r_L d$ |
| **Trainable** | $4 d^2$ | **$4 r_L d$** |

## 5.10 Registry

```python
@ATTENTION_REGISTRY.register("squeezing_heads")
@ATTENTION_REGISTRY.register("squeezing_heads_distiller")
class SqueezingHeadsDistiller(BaseAttention): ...
```

## 5.11 Failure Modes

| Failure | Detection | Response |
|---|---|---|
| Missing weights | `copy_weights` logs | Uses initialized |
| Small Frobenius norm | Warning | Continue |
| Shape mismatch | Truncated copy | Skip layer |
| CUDA OOM | RuntimeError | Skip layer |

---
